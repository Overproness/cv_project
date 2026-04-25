Neresemble:

"""
NeRSemble: Multi-view Radiance Field Reconstruction of Human Heads
Tobias Kirschstein1, Shenhan Qian1, Simon Giebenhain1, Tim Walter1, Matthias Nießner1
1Technical University of Munich
SIGGRAPH 2023

NeRSemble takes multi-view video recordings of a person and allows rendering from arbitrary viewpoints.
RGB Depth Deformations
Abstract
We focus on reconstructing high-fidelity radiance fields of human heads, capturing their animations over time, and synthesizing re-renderings from novel viewpoints at arbitrary time steps.

To this end, we propose a new multi-view capture setup composed of 16 calibrated machine vision cameras that record time-synchronized images at 7.1 MP resolution and 73 frames per second. With our setup, we collect a new dataset of over 4700 high-resolution, high-framerate sequences of more than 220 human heads, from which we introduce a new human head reconstruction benchmark. The recorded sequences cover a wide range of facial dynamics, including head motions, natural expressions, emotions, and spoken language.

In order to reconstruct high-fidelity human heads, we propose Dynamic Neural Radiance Fields using Hash Ensembles (NeRSemble). We represent scene dynamics by combining a deformation field and an ensemble of 3D multi-resolution hash encodings. The deformation field allows for precise modeling of simple scene movements, while the ensemble of hash encodings helps to represent complex dynamics. As a result, we obtain radiance field representations of human heads that capture motion over time and facilitate re-rendering of arbitrary novel viewpoints. In a series of experiments, we explore the design choices of our method and demonstrate that our approach outperforms state-of-the-art dynamic radiance field approaches by a significant margin.

Video

Method Overview

NeRSemble represents a spatio-temporal radiance field for dynamic NVS using volume rendering (left). On the right side, we show how NeRSemble obtains a density 𝜎(x) and color value c(x, d) for a point x on a ray at time 𝑡:

Given the deformation code 𝝎𝑡 the point x is warped to x′ = D(x, 𝝎𝑡 ) in the canonical space.
The resulting point is used to query features H𝑖(x′) from the 𝑖-th hash grid in our ensemble.
The resulting features are blended using weights 𝛽𝑡 . Note that both 𝝎𝑡 and 𝛽𝑡 contribute to explaining temporal changes.
We predict density 𝜎(x) and view-dependent color c(x, d) from the blended features using an efficient rendering head consisting of two small MLPs.
Contributions of Individual Hashtables

𝛽𝑡 =
...
𝛽𝑡,2:

𝛽𝑡,3:

𝛽𝑡,4:

During training, blend weights 𝛽𝑡 are optimized for each timestep 𝑡 to combine the hash tables s.t. they explain the observed expression.

At inference time, choosing different blend weights 𝛽𝑡 allows certain control over the person's expression.

While the first hash table H1 learns a representation similar to the mean face of a person, the remaining hash tables H𝑖 add further expression-dependent details to the scene.

BibTeX
@article{kirschstein2023nersemble,
author = {Kirschstein, Tobias and Qian, Shenhan and Giebenhain, Simon and Walter, Tim and Nie\ss{}ner, Matthias},
title = {NeRSemble: Multi-View Radiance Field Reconstruction of Human Heads},
year = {2023},
issue_date = {August 2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {42},
number = {4},
issn = {0730-0301},
url = {https://doi.org/10.1145/3592455},
doi = {10.1145/3592455},
journal = {ACM Trans. Graph.},
month = {jul},
articleno = {161},
numpages = {14},
}
Website inspired by Keunhong Park's Nerfies website.

Please contact Tobias Kirschstein for feedback and questions.
"""

"""
Paper | Video | Project Page

This is the official download repository for v2 of the NeRSemble dataset. The NeRSemble Dataset is a large-scale multi-view video dataset of facial performances.

1. Overview
   Participant Overview
   static/nersemble_v2_participant_overview.jpg

Camera Overview
static/nersemble_v2_camera_overview.jpg

Expression Overview
static/nersemble_v2_expression_overview.jpg

Statistics
static/nersemble_v2_statistics.jpg

2. Data Access & Setup
   Request access to the NeRSemble dataset: https://forms.gle/rYRoGNh2ed51TDWX9
   Once approved, you will receive a mail with the download link in the form of
   NERSEMBLE_DATA_URL = "..."
   Create a file at ~/.config/nersemble_data/.env with following content:
   NERSEMBLE_DATA_URL = "<<<URL YOU GOT WHEN REQUESTING ACCESS TO NERSEMBLE>>>"
   Install this repository via
   pip install nersemble_data
   Use the download script in this repository to download the parts of the NeRSemble dataset that you need
3. Download Scripts
   Upon installation of the repository with pip, a nersemble-data script is automatically made available that is the main tool for downloading the dataset.
   You can investigate it via:

nersemble-data --help
If for some reason the nersemble-data command cannot be found, you can also invoke the script via

python ./scripts/manage_data.py
from the repository root.

3.1. Get an Overview
nersemble-data list
Lists all participant IDs that are available for download.

nersemble-data list $ID
Lists all available sequences for participant $ID.

3.2. Download data
To download the dataset to your local folder ${nersemble_folder} run:

nersemble-data download ${nersemble_folder}
The script will first summarize all the files to download with an estimate of the total size and ask for confirmation before the actual download happens.
Since the full dataset is more than 1.5 TB large, the script provides several parameters to download only parts of the dataset. Use

nersemble-data download --help
to get a description of each option.
In principle, the dataset contains #PARTICIPANTS x #SEQUENCES x #CAMERAS many videos, and one can select a subset for each dimension to narrow down the download:

--participant: select participant(s) to download
--sequence: select sequence(s) to download
--camera: select camera(s) to download
--n_workers Specify how many downloads should happen in parallel
For example,

nersemble-data download ${nersemble_folder} --participant 240
downloads all videos for participant 240, while

nersemble-data download ${nersemble_folder} --sequence EMO-1-shout+laugh --camera 222200037
would download all participants but only the 222200037 camera for the EMO-1-shout+laugh sequence.

4. Usage
   The repository also comes with a data manager to facilitate loading single images from the downloaded videos:

from nersemble_data.data.nersemble_data import NeRSembleDataManager, NeRSembleParticipantDataManager

nersemble_folder = "path/to/local/nersemble/folder"
data_folder = NeRSembleDataManager(nersemble_folder)
downloaded_participant_ids = data_folder.list_participants() # <- List of all participants that were downloaded
participant_id = downloaded_participant_ids[0] # <- Use first available participant

data_manager = NeRSembleParticipantDataManager(nersemble_folder, participant_id)
downloaded_sequences = data_manager.list_sequences() # <- List of all sequences that were downloaded for that participant
sequence_name = downloaded_sequences[0] # <- Use first available sequence

downloaded_cameras = data_manager.list_cameras(sequence_name) # <- List of all cameras that were downloaded for that sequence
serial = downloaded_cameras[0] # <- Use first available camera
4.1. Load images
timestep = 0 # <- Load first frame of video  
image = data_manager.load_image(sequence_name, serial, timestep)
4.2. Load cameras
camera_calibration = data_manager.load_camera_calibration()
world_2_cam_poses = camera_calibration.world_2_cam # <- For each camera: 4x4 Extrinsic matrices in W2C direction and OpenCV camera coordinate convention
intrinsics = camera_calibration.intrinsics # <- 3x3 intrinsic matrix (shared across all 16 cameras) for 3208x2200 images
4.3. Color Calibration
The v2 of the NeRSemble dataset comes with improved color calibration that improves color consistency across all 16 cameras as well as ensures colors are more realistic in general.
One can apply color calibration already during image loading:

timestep = 0 # <- Load first frame of video  
image = data_manager.load_image(sequence_name, serial, timestep, apply_color_correction=True)
Alternatively, one can load the color correction matrix and apply it to the original image separately:

from nersemble_data.util.color_correction import correct_color

color_calibration = data_manager.load_color_calibration()
ccm = color_calibration[serial]
image_corrected = correct_color(image, ccm)
When using the NeRSemble dataset, please cite the original SIGGRAPH paper:
@article{kirschstein2023nersemble,
author = {Kirschstein, Tobias and Qian, Shenhan and Giebenhain, Simon and Walter, Tim and Nie\ss{}ner, Matthias},
title = {NeRSemble: Multi-View Radiance Field Reconstruction of Human Heads},
year = {2023},
issue_date = {August 2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {42},
number = {4},
issn = {0730-0301},
url = {https://doi.org/10.1145/3592455},
doi = {10.1145/3592455},
journal = {ACM Trans. Graph.},
month = {jul},
articleno = {161},
numpages = {14},
}
Contact Tobias Kirschstein for questions, comments and reporting bugs, or open a GitHub issue.

"""

"""
NeRSemble: Multi-view Radiance Field Reconstruction of Human Heads
Paper | Video | Project Page

Tobias Kirschstein, Shenhan Qian, Simon Giebenhain, Tim Walter and Matthias Nießner
Siggraph 2023

1. Installation
   1.1. Dependencies
   PyTorch 2.0
   nerfstudio
   tinycudann
   Setup environment

conda env create -f environment.yml
conda activate nersemble
which creates a new conda environment nersemble (Installation may take a while).

Manually install tinycudann:

pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
(Also helpful, if you get an error like ImportError: DLL load failed while importing \_86_C: The specified procedure could not be found. later on)

Install the nersemble package itself by running

pip install -e .
inside the cloned repository folder.

1.2. Environment Paths
All paths to data / models / renderings are defined by environment variables.
Please create a file in your home directory in ~/.config/nersemble/.env with the following content:

NERSEMBLE_DATA_PATH="..."
NERSEMBLE_MODELS_PATH="..."
NERSEMBLE_RENDERS_PATH="..."
Replace the ... with the locations where data / models / renderings should be located on your machine.

NERSEMBLE_DATA_PATH: Location of the multi-view video dataset (See section 2 for how to obtain the dataset)
NERSEMBLE_MODELS_PATH: During training, model checkpoints and configs will be saved here
NERSEMBLE_RENDERS_PATH: Video renderings of trained models will be stored here
If you do not like creating a config file in your home directory, you can instead hard-code the paths in the env.py.

1.3. Troubleshooting
You may run into this error at the beginning of training:

\lib\site-packages\torch\include\pybind11\cast.h(624): error: too few arguments for template template parameter "Tuple"
detected during instantiation of class "pybind11::detail::tuple_caster<Tuple, Ts...> [with Tuple=std::pair, Ts=<T1, T2>]"
(721): here

\lib\site-packages\torch\include\pybind11\cast.h(717): error: too few arguments for template template parameter "Tuple"
detected during instantiation of class "pybind11::detail::tuple_caster<Tuple, Ts...> [with Tuple=std::pair, Ts=<T1, T2>]"
(721): here
This occurs during compilation of torch_efficient_distloss and can be solved by either training without distortion loss or by changing one line in the torch_efficient_distloss library (see sunset1995/torch_efficient_distloss#8).

2. Dataset
   Access to the dataset can be requested here.
   To reproduce the experiments from the paper, only download the nersemble_XXX_YYY.zip files (There are 10 in total for the 10 different sequences), as well as the camera_params.zip. Extract these .zip files into NERSEMBLE_DATA_PATH.
   Also, see src/nersemble/data_manager/multi_view_data.py for an explanation of the folder layout.

3. Usage
   3.1. Training
   python scripts/train/train_nersemble.py $ID $SEQUENCE_NAME --name $NAME
   where $ID is the id of the participant in the dataset (e.g., 030) and SEQUENCE_NAME is the name of the expression / emotion / sentence (e.g., EXP-2-eyes). $NAME may optionally be used to annotate the checkpoint folder and the wandb experiment with some descriptive experiment name.

The training script will place model checkpoints and configuration in ${NERSEMBLE_MODELS_PATH}/nersemble/NERS-XXX-${name}/. The incremental run id XXX will be automatically determined.

GPU Requirements
Training takes roughly 1 day and requires at least an RTX A6000 GPU (48GB VRAM). GPU memory requirements may be lowered by tweaking some of these hyperparameters:

--max_n_samples_per_batch: restricts How many ray samples are fed through the model at once (default 20 for 2^20 samples)
--n_hash_encodings: Number of hash encodings in the ensemble (default 32). Using 16 should give comparable quality (--latent_dim_time needs to be set to the same value)
--cone_angle: Use larger steps between ray samples for further away points. The default value of 0 (no step size increase) provides the best quality. Try values up to 0.004
--n_train_rays: Number of rays per batch (default 4096). Lower values can affect convergence
--mlp_num_layers / --mlp_layer_width: Making the deformation field smaller should still provide reasonable performance.
RAM requirements
Per default, the training script will cache loaded images in RAM which can cause RAM usage up to 200G. RAM usage can be lowered by:

--max_cached_images (default 10k): Set to 0 to completely disable caching
Special config for sequences 97 and 124
We disable the occupancy grid acceleration structure from Instant NGP as well as the use of distortion loss due to complex hair motion in sequence 97:

python scripts/train/train_nersemble.sh 97 HAIR --name $name --disable_occupancy_grid --lambda_dist_loss 0
We only train on a subset of sequence 124 (timesteps 95-570) and slightly prolong the warmup phase due to the complexity of the sequence:

python scripts/train/train_nersemble.sh 124 FREE --name $name --start_timestep 95 --n_timesteps 475 --window_hash_encodings_begin 50000 --window_hash_encodings_end 100000
3.2. Evaluation
In the paper, all experiments are conducted by training on only 12 cameras and evaluating rendered images on 4 hold-out views (cameras 222200040, 220700191, 222200043 and 221501007).

For obtaining the reported PSNR, SSIM and LPIPS metrics (evaluated at 15 evenly spaced timesteps):

python scripts/evaluate/evaluate_nersemble.py NERS-XXX
where NERS-XXX is the run name obtained from running the training script above.

For obtaining the JOD video metric (evaluated at 24fps, takes much longer):

python scripts/evaluate/evaluate_nersemble.py NERS-XXX --skip_timesteps 3 --max_eval_timesteps -1
The evaluation results will be printed in the terminal and persisted as a .json file in the model folder ${NERSEMBLE_MODELS_PATH}/NERS-XXX-${name}/evaluation.

3.3. Rendering
From a trained model NERS-XXX, a circular trajectory (4s) may be rendered via:

python scripts/render/render_nersemble.py NERS-XXX
The resulting .mp4 file is stored in NERSEMBLE_RENDERS_PATH.

4. Trained Models
   We provide one trained NeRSemble for each of the 10 sequences used in the paper:

Participant ID Sequence Model
18 EMO-1-shout+laugh NERS-9018
30 EXP-2-eyes NERS-9030
38 EXP-1-head NERS-9038
85 SEN-01-port_strong_smokey NERS-9085
97 HAIR NERS-9097
124 FREE NERS-9124
175 EXP-6-tongue-1 NERS-9175
226 EXP-3-cheeks+nose NERS-9226
227 EXP-5-mouth NERS-9227
240 EXP-4-lips NERS-9240
Simply put the downloaded model folders into ${NERSEMBLE_MODELS_PATH}/nersemble.
You can then use the evaluate_nersemble.py and render_nersemble.py scripts to obtain renderings or reproduce the official metrics below.

5. Official metrics
   Metrics averaged over all 10 sequences from the NVS benchmark (same 10 sequences as in the paper):

Model PSNR SSIM LPIPS JOD
NeRSemble 31.48 0.872 0.217 7.85
Note the following:

The metrics are slightly different from the paper due to the newer version of nerfstudio used in this repository
PSNR, SSIM and LPIPS are computed on only 15 evenly spaced timesteps (to make comparisons cheaper)
JOD is computed on every 3rd timestep (using --skip_timesteps 3 --max_eval_timesteps -1)
Metrics for sequence 97 were computed with --no_use_occupancy_grid_filtering
If you find our code, dataset or paper useful, please consider citing

@article{kirschstein2023nersemble,
author = {Kirschstein, Tobias and Qian, Shenhan and Giebenhain, Simon and Walter, Tim and Nie\ss{}ner, Matthias},
title = {NeRSemble: Multi-View Radiance Field Reconstruction of Human Heads},
year = {2023},
issue_date = {August 2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {42},
number = {4},
issn = {0730-0301},
url = {https://doi.org/10.1145/3592455},
doi = {10.1145/3592455},
journal = {ACM Trans. Graph.},
month = {jul},
articleno = {161},
numpages = {14},
}
Contact Tobias Kirschstein for questions, comments and reporting bugs, or open a GitHub issue.

About
[Siggraph '23] NeRSemble: Neural Radiance Field Reconstruction of Human Heads

tobias-kirschstein.github.io/nersemble/
Topics
avatars nerf 3d-deep-learning 3d-face-reconstruction neural-fields novel-view-synthesis digital-humans dynamic-nerf siggraph2023
Resources
Readme
Activity
Stars
250 stars
Watchers
9 watching
Forks
13 forks
Report repository
Releases
No releases published
Deployments
19
github-pages last year

- 18 deployments
  Packages
  No packages published
  Contributors
  1
  @tobias-kirschstein
  tobias-kirschstein Tobias Kirschstein
  Languages
  Python
  100.0%
  """
