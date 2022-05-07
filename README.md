# 11-785 Introduction to Deep Learning Final Project: 

## Code Organization

Below is the purpose and summarized contents of each Python file in the repository (last updated April 27, 2022)

- "SELDNetwork.py":
  - Baseline model architecture written in PyTorch
  - Note: the repository also contains an earlier one of our group's implementations of SELDNet in the file "seldnet.py"
- "d3net.py": Contains the PyTorch code for D2Blocks, D3Blocks, and RD3Net
- "evaluation.py": Code
  - Note: the repository also contains an earlier F-score and Error metric implementation that could not accomodate Multi-ACCDOA in the file "temp_eval.py"
- "foa_dataset.py": Main DCASE 2022 Dataset Reader and Feature Extractor
- "local_multpath_learning.py": Implementation of Local Multi-path Blocks
- "scale_specific_multipath_learning.py": Implementation of Path-Specific ResBlk
- "loss.py": Implementation of ADPIT-Loss Function
- "SELD_Notebook.ipynb": Jupyter Notebook for running tests

## Citations and Code References

The DCASE 2022 Official [SELDNet Repository](https://github.com/sharathadavanne/seld-dcase2022) was used for baseline reference and testing for our project.

1. Sharath Adavanne, Archontis Politis, Joonas Nikunen and Tuomas Virtanen, "Sound event localization and detection of overlapping sources using convolutional recurrent neural network" in IEEE Journal of Selected Topics in Signal Processing (JSTSP 2018)
2. Kazuki Shimada, Yuichiro Koyama, Shusuke Takahashi, Naoya Takahashi, Emiru Tsunoo, and Yuki Mitsufuji, " Multi-ACCDOA: localizing and detecting overlapping sounds from the same class with auxiliary duplicating permutation invariant training" in the The international Conference on Acoustics, Speech, & Signal Processing (ICASSP 2022)
3. Thi Ngoc Tho Nguyen, Douglas L. Jones, Karn N. Watcharasupat, Huy Phan, and Woon-Seng Gan, "SALSA-Lite: A fast and effective feature for polyphonic sound event localization and detection with microphone arrays" in the International Conference on Acoustics, Speech, & Signal Processing (ICASSP 2022)
