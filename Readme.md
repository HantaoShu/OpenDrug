# Using representer point selection for interpretable drug response prediction (OpenDrug)
 

## About
This directory contains the code and resources of the following paper:

<i>"Using representer point selection for interpretable drug response prediction 
". Under review. </i>


## Overview of the Model


<p align="center">
<img  src="figure/OpenDrug1.png" width="800" height="400" > 
</p>

OpenDrug is a two step interpretable deep learning model for drug response prediction. OpenDrug is a kind of post-hoc
 interpretable deep learning model methods which identify important feature and training example by reanalysis model 
 after 
  whole training process.
### Step 1. Train OpenDrug to predict drug response.
For each drug, we trained a multi-layer neural network model to predict the drug response based on the molecular 
features (gene expression and mutation feature) of each cell
 
### Step 2. Interpreting OpenDrug and identify important training example and feature.





For further details, see Online Methods of our paper. 

## Sub-directories
  - [src] contains implementation of OpenDrug used for the beer review data. 
  - [example_data] contains one example drug data. The drug response data is save as pickle file.
  - [baseline_methods] contains the implementation of baseline methods including elastic net, linear model, random 
  forest, and 
  SVR used
   in comparison.
  - [evaluation] contains the implementation of GCN used in evaluating important training example identified by OpenDrug
  

<br>

## Data
  - We constructed dataset based on Genomics of Drug Sensitivity in Cancer (GDSC) project(Iorio et al., 2016), which 
  measured the drug responses elicited by a panel of 265 drugs on 1,001 tumor cell lines, which can be found and 
  downloaded at the GDSC website (https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources)
  
   - We selected genes from the “mini cancer genome panel” and further removed gene expression features for which the 
   standard 
   deviations fell into the lowest 10% over all the 
   training samples, and gene mutation features with less than 10 somatic mutations across all the training samples.

<br>

## Dependency
- python 3.5
- pytorch 1.5.0
- ogb
- scikit-learn 0.23.2
## Code Usage
1. Train OpenDrug to predict drug response. 
<code>  
cd src \
python train.py --drug_id <drug_id> --data_dir <data_dir> --model_dir <model_dir> \
for example:
python train_LOO.py --drug_id 11 --data_dir ../example_data --model_dir ../model
</code>
    Here, we provide a leave-one-out cross-validation implementation which is convenient for further interpreting model.
    We also provide baseline methods implementation which are compared in our paper. Baseline methods can be trained 
    using 
    following 
    command.\
<code>
cd baseline_methods\
python run_{elastic_net,linear_model,rf,svr}.py  --drug_id <drug_id> --data_dir <processed data path> --model_dir <path
 to save 
model> \
</code>

2. Interpreting OpenDrug and identify important training example and feature.
<code> 
cd src\
python generate_representer_matrix_index.py --drug_id <drug_id> --data_dir <data_dir> --index_list <test_index_list> 
--model_dir <model path> --output_dir <path to save interprate result> 
</code>

    where test_index_list denote the test example index list corresponding to index in processed data pickle. The output 
file is save in numpy format whose weight(W<sub>ij</sub>) represent importance of feature j on training data i.

3. Evaluate  

We identify important training example and features using graph convolutional neural network (GCN). We train a single
 layer GCN model to predict drug response by only using important training example identified by OpenDrug. The 
 important feature identified by OpenDrug can accurately predict drug response and also can be validated by 
 synergistic drug combination data.


## License
DeepDrug is licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0

