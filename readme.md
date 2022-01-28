# Mislabel Detection
This directory contains the data and codes for project 'Detecting Mislabelled Samples in Popular Graph Prediction Benchmarks'.


## File Framework
Ignore directories and files not mentioned.

Directories:
- `dataset`: data directory
- `checkpoints`: to save trained GNNs
- `gnn_results`: to save the prediction results of GNNs
- `mislabel_results`: to save the mislabel detection results
- `tensorboard_logs`: to save the loss info for visualization
- `aum_data`: to save internal results when training AUM

Files:
- `noise_generator.py`: add symmetric or asymmetric noises to a specific dataset (currently support Flickr, Cora, PubMed, CiteSeer)
- `run_GNNs/GNNs_validation.py`: called by mislabel detection methods to train GNNs (currently includes GCN, GIN, GAT)
- `reliability_diagrams.py`: called to draw reliability diagrams
- `run_baseline/Confident_Learning/AUM/NEIGHBORAGG-CMD/DY-bootstrap.py`: code to run different mislabel detection methods
- `run_CL_negsamp.py`: code to run our method
- `evaludate_different_methods.py`: code to compare our method with baselines


## Usage
#### Generate Noise
To add symmetric noise: 

`python noise_generator.py --dataset Flickr --noise_type symmetric --mislabel_rate 0.1`

To add asymmetric noise: 

`python noise_generator.py --dataset Flickr --noise_type asymmetric --mislabel_rate 0.1`

The generated noisy class map will be saved in `./dataset/Flickr/raw`.


#### Run Our Method
`CUDA_VISIBLE_DEVICES=0 python run_CL_negsamp.py --model GCN --noise_type symmetric --mislabel_rate 0.1 --sample_rate 0.5 --classifier MLP --dataset Flickr`


#### Run Baselines
To run a baseline:

`CUDA_VISIBLE_DEVICES=0 python run_baseline.py --model GCN --noise_type symmetric --mislabel_rate 0.1 --dataset Flickr`


#### Evaluation
To evaluate one method (take Confident Learning for example):

`python evaluate_different_methods.py --method CL --model GCN --noise_type symmetric --mislabel_rate 0.1 --dataset Flickr`

To evaluate several methods (concatenate method names with '+'):

`python evaluate_different_methods.py --method baseline+CL+AUM --model GCN --noise_type symmetric --mislabel_rate 0.1 --dataset Flickr`

Currently not support NEIGHBORAGG-CMD because it does not provide a filtering strategy.


## Visualization
To see the loss curve when training GNNs:

`tensorboard --logdir=tensorboard_logs --port=6006`

then ssh to the server and open ` http://localhost:6006/` in local browser.
