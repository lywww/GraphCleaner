# GraphCleaner
This repository contains the codes for paper 'GraphCleaner: Detecting Mislabelled Samples in Popular Graph Prediction Benchmarks'.


## File Framework

Directories:
- `dataset`: data directory
- `checkpoints`: to save trained GNNs
- `gnn_results`: to save the prediction results of GNNs
- `mislabel_results`: to save the mislabel detection results
- `tensorboard_logs`: to save the loss info for visualization
- `aum_data`: to save internal results when training AUM
- `case_studies`: to give the manual judgement files of case study and the two new variants of PubMed dataset

Code Files:
- `noise_generator.py`: download dataset and add symmetric / asymmetric noises to a specific dataset
- `Utils.py`: shared functions called by mislabel detection methods
- `run_GNNs_validation.py`: called by mislabel detection methods to train GNNs
- `run_baseline/Confident_Learning/AUM/DY-bootstrap.py`: code to run different mislabel detection methods
- `run_CL_negsamp.py`: code to run our GraphCleaner
- `run_CL_negsamp_withoutCL.py`: code to run no Cl version of our GraphCleaner
- `evaludate_different_methods.py`: code to evaluate different mislabel detection methods


## Usage
#### Step0: Preparation
`pip install -r requirements.txt`


#### Step1: Generate Noise

`python noise_generator.py --dataset <the name of dataset> --noise_type <symmetric or asymmetric> --mislabel_rate <mislabel rate>`

The `dataset` directory will be created, and datasets will be downloaded automatically. 
This step will generate noisy class map which will be saved under `./dataset/<dataset dir>/raw/` and ground truth mislabel transition matrices.


#### Step2: Run Mislabel Detection Methods
##### Run Our GraphCleaner
`CUDA_VISIBLE_DEVICES=<gpu id> python run_CL_negsamp.py --model <base classifier name> --noise_type <symmetric or asymmetric> --mislabel_rate <mislabel_rate> --dataset <the name of dataset>`

##### Run Other Mislabel Detection Methods
`CUDA_VISIBLE_DEVICES=<gpu id> python run_<method name>.py --model <base classifier name> --noise_type <symmetric or asymmetric> --mislabel_rate <mislabel_rate> --dataset <the name of dataset>`

The `checkpoints`, `gnn_results`, `mislabel_results` and `tensorboard_logs` directories will be created. 
This step will generate the leaned mislabel transition matrices.


#### Step3: Evaluation
Concatenate method names with '+' can evaluate several methods at the same time (`ours` means GraphCleaner):

`python evaluate_different_methods.py --method <example: baseline+CL+AUM> --model <base classifier name> --noise_type <symmetric or asymmetric> --mislabel_rate <mislabel_rate> --dataset <the name of dataset>`


## Visualization
To see the loss curve when training GNNs:

`tensorboard --logdir=tensorboard_logs --port=6006`

then ssh to the server and open ` http://localhost:6006/` in local browser.


## Case Studies
`X_analysis.txt` are manual judgement files for dataset X (X = cora / citeseer / pubmed / arxiv).
`pubmed_clean.pt` is the new dataset `PubMedCleaned`, which removes all zero-labelled and multi-labelled nodes and fixes all detected mislabels.
`pubmed_multi.pt` is the new dataset `PubMedMulti`, which keeps multi-labelled nodes by using binary label vectors, and fixes all detected mislabels.
