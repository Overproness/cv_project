here's readme of FaceScape dataset:

i have also attached an image that they had provided, its the first attached pic.

"""
FaceScape: a Large-scale High Quality 3D Face Dataset and Detailed Riggable 3D Face Prediction

This webpage is the most up-to-date project page providing data access. Please note that the previous website (https://facescape.nju.edu.cn/) has been decommissioned and will no longer be accessible.

Abstract
We present a large-scale detailed 3D face dataset, FaceScape, and propose a novel algorithm that is able to predict elaborate riggable 3D face models from a single image input. FaceScape dataset provides 3D face models, parametric models and multi-view images in large-scale and high-quality. The camera parameters, the age and gender of the subjects are also included. The data have been released to public for non-commercial research purpose.

Dataset
The data available for downloading contains 847 subjects x 20 expressions, in a total of 16,940 models, which is roughly 90% of the complete data. The other 10% of data are not released for potential evaluation or benchmark in the future. The available data includes:

Data Description

1. Information
   Information List (size: 1KB)
   A text file containing the ages and gender of the subjects. From left to right, each row is the index, gender (m-male, f-female), age, and valid label. '-' means this information is not provided. Valid label is [1 + 4] binary number, 1-True, 0-False. The first number means if the model for this person is complete and valid, and the rest four means if obj-model, mtl-material, jpg-texture, and png-dpmap are missing.
   Publishable List (size: 1KB)
   A text file containing the indexes of the model that can be used for paper publication or presentation. Please read the 4th term in the license for more about this policy. The publishable list may be updated in the future.
2. TU Models (size: 120GB)
   There are 847 tuple of topologically uniformed models. Each tuple of data consists of:

20 base mesh models (/models*reg/$IDENTITY$*$EXPRESSION$.obj)
20 displacement maps (/dpmap/$IDENTITY$_$EXPRESSION$.png)
1 base material file (/models_reg/$IDENTITY$_$EXPRESSION$.obj.mtl)
1 texture (/models*reg/$IDENTITY$*$EXPRESSION$.jpg) where $IDENTITY$ is the index of identity (1 - 847), $EXPRESSION$ is the index of expression (0 - 20). Please note that some of the model's texture maps (index: 360 - 847) were mosaics around the eyes to protect the privacy of some participants.

3. Multi-view Data
   FaceScape provides multi-view images, camera paramters and reconstructed 3D shapes. There are 359 subjects x 20 expressions = 7120 tuples of data. The number of available images reaches to over 400k.

Please view here for detailed description and usage of the multi-view data.

4. Bilinear model (size: 4.67GB)
   Our bilinear model is a statistical model which transforms the base shape of the faces into a vector space representation. We provide two 3DMM with different numbers of identity parameters:

core_847_50_52.npy - bilinear model with 52 expression parameters and 50 identity parameters.
core_847_300_52.npy - bilinear model with 52 expression parameters and 300 identity parameters.
factors_id_847_50_52.npy and factors_id_847_300_52.npy are identity parameters corresponding to 847 subjects in the dataset.
Please see here for the usage and the demo code.

5. Tools
   We provide Python code to extract facial landmarks and facial region from the TU-models. Please keep a watch on our project page where the latest resources will be updated in the future.

Preview
One sample is rendered online as shown below. The online-rendered model is the down-sampled version of provided model, because high-resolution displacement map is too slow to be rendered online. The rendering result with the high-resolution displacement map is shown in the figure below the online-renderer.

Online Rendering (Down-Sampled)

Offline Rendering

factors_id_847_50_52.npy and factors_id_847_300_52.npy are identity parameters corresponding to 847 subjects in the dataset.
Features

1. Topologically uniformed.
   The geometric models of different identities and different expressions share the same mesh topology, which makes the features on faces easy to be aligned. This also helps in building a 3D morphable model.

2. Displacement map + base mesh.
   We use base shapes to represent rough geometry and displacement maps to represent detailed geometry, which is a two-layer representation for our extremely detailed face shape. Some light-weight software like MeshLab can only visualize the base mesh model/texture. Displacement maps can be loaded and visualized in MAYA, ZBrush, 3D MAX, etc.

3. 20 specific expressions.
   The subjects are asked to perform 20 specific expressions for capturing: neutral, smile, mouth-stretch, anger, jaw-left, jaw-right, jaw-forward, mouth-left, mouth-right, dimpler, chin-raiser, lip-puckerer, lip-funneler, sadness, lip-roll, grin, cheek-blowing, eye-closed, brow-raiser, brow-lower.

4. High resolution.
   The texture maps and displacement maps reach 4K resolution, which preserving maximum detailed texture and geometry.

Data Access
For downloading the dataset, please complete the License Agreement and send it to nju3dv@nju.edu.cn, and download from download link (Google Drive) or download link (Baidu Netdisk).

When you submit request, which means you have read, understand, and commit to the entirety of the License Agreement. There are, still, a few KEY POINTS which need to emphasise again:
The email subject should be [FaceScape Dataset Request].
NO COMMERCIAL USE: The license granted is for internal, non-commercial research, evaluation or testing purposes only. Any use of the DATA or its contents to manufacture or sell products or technologies (or portions thereof) either directly or indirectly for any direct or indirect for-profit purposes is strictly prohibited.
NO WARRANTY: The data are provided "as is" and any express or implied warranties are disclaimed.
RESTRICTED USE IN RESEARCH: The portraits including images and rendered model cannot be published in any form, except for the data as listed in the publishable list.
FAQ

1. How can an undergraduate or graduate student get access to the data?
2. Can I use the data in paper publication or presentation?
3. Why using displacement map?
4. Can I transform a "displacement map + base mesh" model to a high-vertex mesh?
5. Why are some textures blurry around the eyes?
6. How to extract the facial region from the whole head?
   Contact Information
   If you have some questions, please refer to the issue section of our GitHub repository, or send email to nju3dv@nju.edu.cn and cc zh@nju.edu.cn. We recommend to firstly browse the FAQ and the solved issues in GitHub repository, where the answer you want may has been given.

(back to top)

BibTeX
@article{zhu2023facescape,
title={Facescape: 3d facial dataset and benchmark for single-view 3d face reconstruction},
author={Zhu, Hao and Yang, Haotian and Guo, Longwei and Zhang, Yidi and Wang, Yanru and Huang, Mingkai and Wu, Menghua and Shen, Qiu and Yang, Ruigang and Cao, Xun},
journal={IEEE transactions on pattern analysis and machine intelligence},
volume={45},
number={12},
pages={14528--14545},
year={2023},
publisher={IEEE}
}
@inproceedings{yang2020facescape,
author={Yang, Haotian and Zhu, Hao, Wang, Yanru and Huang, Mingkai and Shen, Qiu and Yang, Ruigang and Cao, Xun},
title={FaceScape: A Large-Scale High Quality 3D Face Dataset and Detailed Riggable 3D Face Prediction},
booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month={June},
year={2020},
page={601--610}
}
"""

"""
FaceScape
FaceScape provides large-scale high-quality 3D face datasets, parametric models, docs and toolkits about 3D face related technology. [CVPR2020 paper] [extended arXiv Report] [supplementary]

Our latest progress will be updated to this repository constantly - [latest update: 2026/01/27]

Data
New: The data can be accessed at the new website https://nju-3dv.github.io/projects/FaceScape/. The old website (facescape.nju.edu.cn) will be decommissioned soon.

The available sources include:

Item (Docs) Description Quantity Quality
TU models Topologically uniformed 3D face models
with displacement map and texture map. 16940 models
(847 id × 20 exp) Detailed geometry,
4K dp/tex maps
Multi-view data Multi-view images, camera parameters
and corresponding 3D face mesh. >400k images
(359 id × 20 exp
× ≈60 view) 4M~12M pixels
Bilinear model The statistical model to transform the base
shape into the vector space. 4 for different settings Only for base shape.
Info list Gender / age of the subjects. 847 subjects --
The datasets are only released for non-commercial research use. As facial data involves the privacy of participants, we use strict license terms to ensure that the dataset is not abused.

Benchmark for SVFR
We present a benchmark to evaluate the accuracy of single-view face 3D reconstruction (SVFR) methods, view here for the details.

ToolKit
Start using python toolkit here, the demos include:

bilinear_model-basic - use facescape bilinear model to generate 3D mesh models.
bilinear_model-fit - fit the bilinear model to 2D/3D landmarks.
multi-view-project - Project 3D models to multi-view images.
landmark - extract landmarks using predefined vertex index.
facial_mask - extract facial region from the full head TU-models.
render - render TU-models to color images and depth map.
alignment - align all the multi-view models.
symmetry - get the correspondence of the vertices on TU-models from left side to right side.
rig - rig 20 expressions to 52 expressions.
Our More Projects related to FaceScape
Towards Native Generative Model for 3D Head Avatar (Fundamental Research 2026)
Yiyu Zhuang*, Hao Zhu*, Jiawei Zhang*, Yuxiao He*, Yanwen Wang, Jiahe Zhu, Yao Yao, Siyu Zhu, Xun Cao#

FATE: Full-head Gaussian Avatar with Textural Editing from Monocular Video (CVPR 2025)
Jiawei Zhang, Zijian Wu, Zhiyang Liang, Yicheng Gong, Dongfang Hu, Yao Yao, Xun Cao, Hao Zhu#

DicFace: Dirichlet-Constrained Variational Codebook Learning for Temporally Coherent Video Face Restoration (CVPR 2025)
Yan Chen*, Hanlin Shang*, Ce Liu, Yuxuan Chen, Hui Li, Weihao Yuan, Hao Zhu, Zilong Dong, Siyu Zhu#

VividTalk: One-Shot Audio-Driven Talking Head Generation Based on 3D Hybrid Prior (3DV 2025)
Xusen Sun, Longhao Zhang, Hao Zhu#, Peng Zhang#, Bang Zhang, Xinya Ji, Kangneng Zhou, Daiheng Gao, Liefeng Bo, Xun Cao

Hallo2: Long-Duration and High-Resolution Audio-Driven Portrait Image Animation (ICLR 2025)
Jiahao Cui*, Hui Li*, Yao Yao, Hao Zhu, Hanlin Shang, Kaihui Cheng, Hang Zhou, Siyu Zhu#, Jingdong Wang

EmoTalk3D: High-Fidelity Free-View Synthesis of Emotional 3D Talking Head (ECCV 2024)
Qianyun He, Xinya Ji, Yicheng Gong, Yuanxun Lu, Zhengyu Diao, Linjia Huang, Yao Yao, Siyu Zhu, Zhan Ma, Songcen Xu, Xiaofei Wu, Zixiao Zhang, Xun Cao, Hao Zhu#

Head360: Learning a Parametric 3D Full-Head for Free-View Synthesis in 360° (ECCV 2024)
Yuxiao He, Yiyu Zhuang, Yanwen Wang, Yao Yao, Siyu Zhu, Xiaoyu Li, Qi Zhang, Xun Cao, Hao Zhu#

High-fidelity 3D Face Generation from Natural Language Descriptions (CVPR 2023)
Menghua Wu, Hao Zhu#, Linjia Huang, Yiyu Zhuang, Yuanxun Lu, Xun Cao

RAFaRe: Learning Robust and Accurate Non-parametric 3D Face Reconstruction from Pseudo 2D&3D Pairs (AAAI 2023)
Longwei Guo, Hao Zhu#, Yuanxun Lu, Menghua Wu, Xun Cao

Detailed Facial Geometry Recovery from Multi-view Images by Learning an Implicit Function (AAAI 2022)
Yunze Xiao*, Hao Zhu*, Haotian Yang, Zhengyu Diao, Xiangju Lu, Xun Cao

ChangeLog
2026/01/27
The download website for the FaceScape dataset has been relocated to https://nju-3dv.github.io/projects/FaceScape/. All data can now be accessed on the new site.
2023/10/20
Benchmark data and results have been updated to be consistent with the experiments in the latest journal version paper.
2022/9/9
One section is added to introduce open-source projects that use FaceScape data or models, and will be continuously updated.
2022/7/26
The data for training and testing MoFaNeRF is added to the download page.
2021/12/2
A benchmark to evaluate single-view face reconstruction is available, view here for detail.
2021/8/16
Share link on Google Drive is available after requesting the license key, view here for details.
2021/5/13
The fitting demo is added to the toolkit. Please note if you downloaded the bilinear model v1.6 before 2021/5/13, you need to download it again, because some parameters required by the fitting demo are supplemented.
2021/4/14
The bilinear model has been updated to 1.6, check it here.
The new bilinear model can now be downloaded from NJU Drive or Google Drive without requesting a license key. Check it here.
ToolKit and Doc have been updated with new content.
Some wrong ages and genders in the info list are corrected in "info_list_v2.txt".
2020/9/27
The code of detailed riggable 3D face prediction is released, check it here.
2020/7/25
Multi-view data is available for download.
The bilinear model is updated to ver 1.3, with vertex-color added.
Info list including gender and age is available on the download page.
Tools and samples are added to this repository.
2020/7/7
The bilinear model is updated to ver 1.2.
2020/6/13
The website of FaceScape is online.
3D models and bilinear models are available for download.
2020/3/31
The pre-print paper is available on arXiv.
Bibtex
If you find this project helpful to your research, please consider citing:

@article{zhu2023facescape,
title={FaceScape: 3D Facial Dataset and Benchmark for Single-View 3D Face Reconstruction},
author={Zhu, Hao and Yang, Haotian and Guo, Longwei and Zhang, Yidi and Wang, Yanru and Huang, Mingkai and Wu, Menghua and Shen, Qiu and Yang, Ruigang and Cao, Xun},
journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
year={2023},
publisher={IEEE}}
@inproceedings{yang2020facescape,
author = {Yang, Haotian and Zhu, Hao and Wang, Yanru and Huang, Mingkai and Shen, Qiu and Yang, Ruigang and Cao, Xun},
title = {FaceScape: A Large-Scale High Quality 3D Face Dataset and Detailed Riggable 3D Face Prediction},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020},
page = {601--610}}
Acknowledge
The project is supported by CITE Lab of Nanjing University, Baidu Research, and Aiqiyi Inc. The student contributors: Shengyu Ji, Wei Jin, Mingkai Huang, Yanru Wang, Haotian Yang, Yidi Zhang, Yunze Xiao, Yuxin Ding, Longwei Guo, Menghua Wu, Yiyu Zhuang.
"""
