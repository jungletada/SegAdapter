# Simple and Efficient Vision Backbone Adapter for Image Semantic Segmentation
Dingjie Peng and Wataru kameyama, Waseda University

 *Our code is based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) for implementation.*



## Get started: install and run mmseg
Please refer to the document of [MMSegmentation](https://mmsegmentation.readthedocs.io/en/latest/index.html) and install the [mmsegmentation-1.0](https://github.com/open-mmlab/mmsegmentation).

Prerequisites: In this section we demonstrate how to prepare an environment with PyTorch.
MMSegmentation works on Linux, Windows and macOS. It requires Python 3.7+, CUDA 10.2+ and PyTorch 1.8+.

Note: If you are experienced with PyTorch and have already installed it, just skip this part and jump to the next section. Otherwise, you can follow these steps for the preparation.

1. Download and install Miniconda from the official website.
2. Create a conda environment and activate it.
3. Install PyTorch following official instructions.
4. Install MMCV using MIM and MMSegmentation.

## How to use the code
1. Verify whether MMSegmentation is installed correctly.
2. download the dataset (ADE20K, COCO-Stuff, Pascal VOC) accoring to MMSegmentation guidelines.
2. Simply replace the directory **demo**, **mmseg**, **tools** and **configs** in your mmsegmentation directory. The model files, configurations are included in **mmseg** and **configs**, respcetively. You can run the files **image_demo.py** in **demo** for visualization.
3. Create a directory and rename it as **checkpoints** in your mmsegmentation directory.
3. The training, testing and evaluation are strictly followed the principle in mmsegmentation-1.0. Please refer to **tools**.

