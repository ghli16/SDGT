# Structure-based virulence factor classification using a dual-driven graph transformer with a pretrained language model


## Environment Requirement
The code has been tested running under Python 3.8.18. The required packages are as follows
* python == 3.7.13
* pandas == 2.0.3
* numpy == 1.24.3
* pyg == 2.4.0
* pytorch-cuda == 11.8
* pytorch == 2.1.1
* scipy == 1.10.1


## Files:
1. Dataset
   - VFDB_fasta:Virulence factor sequences
2. src
   + Model.py: SDGT model framework;
   + self_attention_pooling.py: self attention pooling layer;
   + main.py: Model training validation and testing documentation;
   + utils: Additional tool functions and hyperparameter settings for the model.

