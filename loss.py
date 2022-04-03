import torch
import torch.nn as nn
from torch.nn.functional import mse_loss

import numpy as np
from itertools import permutations


class PITLoss(nn.Module):
    """
    Loss Function Between Two Multi-ACCDOA Vectors

    See also: https://arxiv.org/pdf/2110.07124.pdf
    See also: https://github.com/sharathadavanne/seld-dcase2022
    """

    def __init__(self, n=3):
        """
        Init Function for PITLoss
        :param n: Maximum Number of Simultaneous Sounds from the Same Class
        """

        super().__init__()

        # Calculate All Permutations of n Unique Elements and Store it In a Numpy Array
        index_range = np.arange(n)
        self.possible_permutations = np.array(list(set(permutations(index_range))))

    def forward(self, x, y):
        """
        Loss-Calculating Function for PITLoss
        :param x: Estimated Multi-ACCDOA
        :param y: True Multi-ACCDOA
        :return: PIT Loss
        """

        # Index Through All Possible Permutations of N (Creates A New Axis for Each Permutation)
        accdoa_perms = x[:, self.possible_permutations, :, :]

        # Expand the True Vector in the Same Way so the Sizes are the Same
        new_size = [-1 if i != 1 else len(self.possible_permutations) for i in range(y.dim()+1)]
        expanded_labels = torch.unsqueeze(y, 1).expand(*new_size)

        # Calculate the Best MSE Loss for Each ACCDOA Permutation
        mse_perms = mse_loss(accdoa_perms, expanded_labels, reduction='none')  # Element-Wise Loss
        mse_perms_average = torch.mean(mse_perms, dim=[2, 3])  # Average Loss Over N and 3D Dimension
        best_mse_perms = torch.min(mse_perms_average, dim=1)  # Find the Min Loss Among Permutations

        # Average Min Loss over Classes TODO: Should Time Dimension Be Here?
        pit_loss = torch.mean(best_mse_perms.values, [-1-i for i in range(len(best_mse_perms.values.shape)-1)])
        pit_loss_term = torch.mean(pit_loss)  # Average over Batch (Should this be sum?)
        return pit_loss_term
