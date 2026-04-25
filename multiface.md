here's the readme of multiface dataset:

https://github.com/facebookresearch/multiface

i don't think we need such high resolution.
"""
Multiface Dataset
Our dataset consists of high quality recordings of the faces of 13 identities, each captured in a multi-view capture stage performing various facial expressions. An average of 12,200 (v1 scripts) to 23,000 (v2 scripts) frames per subject with capture rate at 30 fps. Each frame includes roughly 40 (v1) to 160 (v2) different camera views under uniform illumination, yielding a total dataset size of 65TB. We provide the raw captured images from each camera view at a resolution of 2048 × 1334 pixels, tracked meshes including headposes, unwrapped textures at 1024 × 1024 pixels, metadata including intrinsic and extrinsic camera calibrations, and audio. This repository hosts the code of downloading the dataset and building a Codec Avatar using a deep appearance model. To learn more about how the dataset is captured and how different model architectures can influence performance, you may refer to our Technical Report.

Contents
Features
Installation
Quick Start
Works Using this Dataset
Contributors
Citation
License
Features
Comprehensive capture of a wide range of facial expressions
High-quality tracked mesh for each frame
High-resolution (2k), multi-view captured images
6 assets are provided: raw images, unwrapped textures, tracked meshes, headpose, audio and metadata.
Installation
Quick Data Exploration
To download our data, first clone this repository and install dependencies

git clone https://github.com/facebookresearch/multiface
cd multiface
pip3 install -r requirements.txt
Since the full dataset takes terabytes of storage, one may wish to download partially. If you want to view the example assets, you may download the mini-dataset (16.2 GB)

python3 download_dataset.py --dest "/path/to/mini_dataset/" --download_config "./mini_download_config.json"
The download_config argument points to the configuration file specifying assets to be downloaded, options include:

Variable Type Default
entity list of string All the entity will be downloaded
image boolean Raw images of entities selected will be downloaded
mesh boolean Tracked mesh of entities selected will be downloaded
texture boolean Unwrapped texture of entities selected will be downloaded
metadata boolean Metadata of entities selected will be downloaded
audio boolean Audio of entities selected will be downloaded. The first available frame is aligned with the start of the audio file, however, missing frames in images need to be handled for alignment
expression list of string All the facial expression (contains both v1 and v2 scripts) will be downloaded
The configuration to download all assets can be found at download_config.json.

Full Installation
To run training and render the 3D faces, please refer to our full installation guide.

Quick Start
To learn more on selecting model architecture, camera split and expression split for training and testing set, please refer to quick start.

Works Using this Dataset
Deep Appearance Models For Face Rendering Learning Compositional Radiance Fields of Dynamic Human Heads Pixel Codec Avatars
MeshTalk: 3D Face Animation from Speech using Cross-Modality Disentanglement Deep Incremental Learning for Efficient High-Fidelity Face Tracking Modeling Facial Geometry using Compositional VAEs
Strand-accurate Multi-view Hair Capture Mixture of Volumetric Primitives for Efficient Neural Rendering [Code Available] Human Hair Inverse Rendering using Multi-View Photometric data
Contributors
Thanks to all the people who has helped generate and maintain this dataset!

Citation
If you use any data from this dataset or any code released in this repository, please cite the technical report (https://arxiv.org/abs/2207.11243)

@inproceedings{wuu2022multiface,
title={Multiface: A Dataset for Neural Face Rendering},
author = {Wuu, Cheng-hsin and Zheng, Ningyuan and Ardisson, Scott and Bali, Rohan and Belko, Danielle and Brockmeyer, Eric and Evans, Lucas and Godisart, Timothy and Ha, Hyowon and Huang, Xuhua and Hypes, Alexander and Koska, Taylor and Krenn, Steven and Lombardi, Stephen and Luo, Xiaomin and McPhail, Kevyn and Millerschoen, Laura and Perdoch, Michal and Pitts, Mark and Richard, Alexander and Saragih, Jason and Saragih, Junko and Shiratori, Takaaki and Simon, Tomas and Stewart, Matt and Trimble, Autumn and Weng, Xinshuo and Whitewolf, David and Wu, Chenglei and Yu, Shoou-I and Sheikh, Yaser},
booktitle={arXiv},
year={2022},
doi = {10.48550/ARXIV.2207.11243},
url = {https://arxiv.org/abs/2207.11243}
}
License
Multiface is CC-BY-NC 4.0 licensed, as found in the LICENSE file.

[Terms of Use] [Privacy Policy]

"""
