import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import ConvBlock, _ResNet, _ResnetBasicBlock

import fire
import logging
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from utils import build_database, build_datamodule
from utils import manage_experiments
from utils import MyLoggingCallback
from model_utils import init_layer, init_gru, PositionalEncoding

class BaseEncoder(nn.Module):
    def __init__(
        self,
        n_input_channels: int = 1,
        n_output_channels: int = 512,
        p_dropout: float = 0.0,
        time_downsample_ratio: int = 16,
        **kwargs
    ):
        super().__init__()
        self.n_input_channels = n_input_channels
        self.p_dropout = p_dropout
        self.n_output_channels = n_output_channels
        self.time_downsample_ratio = time_downsample_ratio


class PannResNet22(BaseEncoder):
    """
    Derived from PANN ResNet22 network. PannResNet22L17 has 4 basic resnet blocks
    """

    def __init__(self, n_input_channels: int = 1, p_dropout: float = 0.0, **kwargs):
        """
        :param n_input_channels: Number of input channels.
        :param p_dropout: Dropout probability.
        :param pretrained: If True, load pretrained model.
        """
        super().__init__(
            n_input_channels=n_input_channels,
            n_output_channels=512,
            p_dropout=p_dropout,
            time_downsample_ratio=16,
        )

        self.conv_block1 = ConvBlock(in_channels=n_input_channels, out_channels=64)

        self.resnet = _ResNet(block=_ResnetBasicBlock, layers=[2, 2, 2, 2], zero_init_residual=True)

    def forward(self, x):
        """Input: Input x: (batch_size, n_channels, n_timesteps, n_features)"""
        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=self.p_dropout, training=self.training, inplace=True)
        x = self.resnet(x)

        return x

class SeldDecoder(nn.Module):
    """
    Decoder for SELD.
    input: batch_size x n_frames x input_size
    """
    def __init__(self, n_output_channels, n_classes: int = 13, output_format: str = 'reg_xyz',
                 decoder_type: str = None, freq_pool: str = None, decoder_size: int = 128, **kwargs):
        super().__init__()
        self.n_classes = n_classes
        self.decoder_type = decoder_type
        self.freq_pool = freq_pool
        self.doa_format = output_format

        logger = logging.getLogger('lightning')
        logger.info('Map decoder type: {}'.format(self.decoder_type))
        assert self.decoder_type in ['gru', 'bigru', 'lstm', 'bilstm', 'transformer'], \
            'Invalid decoder type {}'.format(self.decoder_type)

        if self.decoder_type == 'gru':
            self.gru_input_size = n_output_channels
            self.gru_size = decoder_size
            self.fc_size = self.gru_size

            self.gru = nn.GRU(input_size=self.gru_input_size, hidden_size=self.gru_size,
                              num_layers=2, batch_first=True, bidirectional=False, dropout=0.3)
            init_gru(self.gru)
        elif self.decoder_type == 'bigru':
            self.gru_input_size = n_output_channels
            self.gru_size = decoder_size
            self.fc_size = self.gru_size * 2

            self.gru = nn.GRU(input_size=self.gru_input_size, hidden_size=self.gru_size,
                              num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
            init_gru(self.gru)
        elif self.decoder_type == 'lstm':
            self.lstm_input_size = n_output_channels
            self.lstm_size = decoder_size
            self.fc_size = self.lstm_size

            self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.gru_size,
                                num_layers=2, batch_first=True, bidirectional=False, dropout=0.3)
            init_gru(self.lstm)
        elif self.decoder_type == 'bilstm':
            self.lstm_input_size = n_output_channels
            self.lstm_size = decoder_size
            self.fc_size = self.lstm_size * 2

            self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.gru_size,
                               num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
            init_gru(self.lstm)
        elif self.decoder_type == 'transformer':
            dim_feedforward = 1024
            self.decoder_input_size = n_output_channels
            self.fc_size = self.decoder_input_size
            self.pe = PositionalEncoding(pos_len=2000, d_model=self.decoder_input_size, dropout=0.0)
            encoder_layer = nn.TransformerEncoderLayer(d_model=self.decoder_input_size,
                                                       dim_feedforward=dim_feedforward, nhead=8, dropout=0.2)
            self.decoder_layer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        else:
            raise NotImplementedError('decoder type: {} is not implemented'.format(self.decoder_type))

        # sed
        self.event_fc_1 = nn.Linear(self.fc_size, self.fc_size // 2, bias=True)
        self.event_dropout_1 = nn.Dropout(p=0.2)
        self.event_fc_2 = nn.Linear(self.fc_size//2, self.n_classes, bias=True)
        self.event_dropout_2 = nn.Dropout(p=0.2)

        # doa
        self.x_fc_1 = nn.Linear(self.fc_size, self.fc_size//2, bias=True)
        self.y_fc_1 = nn.Linear(self.fc_size, self.fc_size//2, bias=True)
        self.z_fc_1 = nn.Linear(self.fc_size, self.fc_size//2, bias=True)
        self.x_dropout_1 = nn.Dropout(p=0.2)
        self.y_dropout_1 = nn.Dropout(p=0.2)
        self.z_dropout_1 = nn.Dropout(p=0.2)
        self.x_fc_2 = nn.Linear(self.fc_size//2, self.n_classes, bias=True)
        self.y_fc_2 = nn.Linear(self.fc_size//2, self.n_classes, bias=True)
        self.z_fc_2 = nn.Linear(self.fc_size//2, self.n_classes, bias=True)
        self.x_dropout_2 = nn.Dropout(p=0.2)
        self.y_dropout_2 = nn.Dropout(p=0.2)
        self.z_dropout_2 = nn.Dropout(p=0.2)

        self.init_weights()

    def init_weights(self):
        init_layer(self.event_fc_1)
        init_layer(self.event_fc_2)
        init_layer(self.x_fc_1)
        init_layer(self.y_fc_1)
        init_layer(self.z_fc_1)
        init_layer(self.x_fc_2)
        init_layer(self.y_fc_2)
        init_layer(self.z_fc_2)

    def forward(self, x):
        """
        :params x: (batch_size, n_channels, n_timesteps/n_frames (downsampled), n_features/n_freqs (downsampled)
        """
        if self.freq_pool == 'avg':
            x = torch.mean(x, dim=3)
        elif self.freq_pool == 'max':
            (x, _) = torch.max(x, dim=3)
        elif self.freq_pool == 'avg_max':
            x1 = torch.mean(x, dim=3)
            (x, _) = torch.max(x, dim=3)
            x = x1 + x
        else:
            raise NotImplementedError('freq pooling {} is not implemented'.format(self.freq_pool))
        '''(batch_size, feature_maps, time_steps)'''

        # swap dimension: batch_size, n_timesteps, n_channels/n_features
        x = x.transpose(1, 2)

        if self.decoder_type in ['gru', 'bigru']:
            x, _ = self.gru(x)
        elif self.decoder_type in ['lsmt', 'bilstm']:
            x, _ = self.lstm(x)
        elif self.decoder_type == 'transformer':
            x = x.transpose(1, 2)  # undo swap: batch size,  n_features, n_timesteps,
            x = self.pe(x)  # batch_size, n_channels/n features, n_timesteps
            x = x.permute(2, 0, 1)  # T (n_timesteps), N (batch_size), C (n_features)
            x = self.decoder_layer(x)
            x = x.permute(1, 0, 2)  # batch_size, n_timesteps, n_features

        # SED: multi-label multi-class classification, without sigmoid
        event_frame_logit = F.relu_(self.event_fc_1(self.event_dropout_1(x)))  # (batch_size, time_steps, n_classes)
        event_frame_logit = self.event_fc_2(self.event_dropout_2(event_frame_logit))

        # DOA: regression
        x_output = F.relu_(self.x_fc_1(self.x_dropout_1(x)))
        x_output = torch.tanh(self.x_fc_2(self.x_dropout_2(x_output)))
        y_output = F.relu_(self.y_fc_1(self.y_dropout_1(x)))
        y_output = torch.tanh(self.y_fc_2(self.y_dropout_2(y_output)))
        z_output = F.relu_(self.z_fc_1(self.z_dropout_1(x)))
        z_output = torch.tanh(self.z_fc_2(self.z_dropout_2(z_output)))
        doa_output = torch.cat((x_output, y_output, z_output), dim=-1)  # (batch_size, time_steps, 3 * n_classes)

        output = {
            'event_frame_logit': event_frame_logit,
            'doa_frame_output': doa_output,
        }

        return output

"""

"""


"""
train
"""

def train(exp_config: str = './configs/seld.yml',
          exp_group_dir: str = '/media/tho_nguyen/disk2/new_seld/dcase2021/outputs',
          exp_suffix: str = '_test',
          resume: bool = False):
    # Load config, create folders, logging
    cfg = manage_experiments(exp_config=exp_config, exp_group_dir=exp_group_dir, exp_suffix=exp_suffix, is_train=True)
    logger = logging.getLogger('lightning')
    # pl.seed_everything(cfg.seed) # Set random seed for reproducible

    
    resume_from_checkpoint = None

    # Load feature database
    feature_db = build_database(cfg=cfg)

    # Load data module
    datamodule = build_datamodule(cfg=cfg, feature_db=feature_db)
    datamodule.setup(stage='fit')
    steps_per_train_epoch = int(len(datamodule.train_dataloader()) * cfg.data.train_fraction)

    # Set learning params
    lr_scheduler = LearningRateScheduler(steps_per_epoch=steps_per_train_epoch, max_epochs=cfg.training.max_epochs,
                                         milestones=cfg.training.lr_scheduler.milestones,
                                         lrs=cfg.training.lr_scheduler.lrs, moms=cfg.training.lr_scheduler.moms)
    logger.info('Finish configuring learning rate scheduler.')

    # # Model checkpoint
    model_checkpoint = ModelCheckpoint(dirpath=cfg.dir.model.checkpoint, filename='{epoch:03d}')  # also save last model
    save_best_model = ModelCheckpoint(monitor='valSeld', mode='min', period=cfg.training.val_interval,
                                      dirpath=cfg.dir.model.best, save_top_k=1,
                                      filename='{epoch:03d}-{valSeld:.3f}-{valER:.3f}-{valF1:.3f}-{valLE:.3f}-'
                                               '{valLR:.3f}')

    # Tensorboard logger
    tb_logger = TensorBoardLogger(save_dir=cfg.dir.tb_dir, name='my_model')

    # Build encoder and decoder
    encoder_params = cfg.model.encoder.__dict__
    encoder = build_model(**encoder_params)
    decoder_params = cfg.model.decoder.__dict__
    decoder_params = {'n_output_channels': encoder.n_output_channels, 'n_classes': cfg.data.n_classes,
                      'output_format': cfg.data.output_format, **decoder_params}
    decoder = build_model(**decoder_params)

    # Build Lightning model
    submission_dir = os.path.join(cfg.dir.output_dir.submission, '_temp')  # to temporarily store val output
    os.makedirs(submission_dir, exist_ok=True)
    model = build_task(encoder=encoder, decoder=decoder, cfg=cfg, submission_dir=submission_dir,
                       test_chunk_len=feature_db.test_chunk_len, test_chunk_hop_len=feature_db.test_chunk_hop_len)

    # Train
    callback_list = [lr_scheduler, console_logger, model_checkpoint]
    if cfg.mode == 'crossval':
        callback_list.append(save_best_model)
        max_epochs = cfg.training.max_epochs
    elif cfg.mode == 'eval':
        max_epochs = cfg.training.best_epoch
    else:
        raise ValueError('Invalid mode {}'.format(cfg.mode))
    #
    trainer = pl.Trainer(gpus=torch.cuda.device_count(), resume_from_checkpoint=resume_from_checkpoint,
                         max_epochs=max_epochs, logger=tb_logger, progress_bar_refresh_rate=2,
                         check_val_every_n_epoch=cfg.training.val_interval,
                         log_every_n_steps=100, flush_logs_every_n_steps=200,
                         limit_train_batches=cfg.data.train_fraction, limit_val_batches=cfg.data.val_fraction,
                         callbacks=callback_list)
    trainer.fit(model, datamodule)
    if cfg.mode == 'crossval':
        logger.info('Best model checkpoint: {}'.format(save_best_model.best_model_path))

    # Test: lightning default takes the best model
    trainer.test()


if __name__ == '__main__':
    fire.Fire(train)
