# Gaussian-Smoothing-in-Saliency-Maps

`two_split.py` is for training two neural networks on two disjoint splits of the training set.

`saliency_diff_plot_sigma.py` is to draw the figure of stability error with different sigmas.

`fidelity_plot_sigma.py` is to draw the figure of fidelity error with different sigmas.

`saliency_diff.py` is to calculate the metrics of SSIM and top-k mIoU.

If you find this code useful, please consider citing our paper:

@inproceedings{yetrade,
  title={On the Trade-Off between Stability and Fidelity of Gaussian-Smoothed Saliency Maps},
  author={Ye, Zhuorui and Farnia, Farzan},
  booktitle={The 28th International Conference on Artificial Intelligence and Statistics}
}
