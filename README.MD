# Using representer point selection for interpretable drug response prediction (OpenDrug)

## About
This directory contains the code and resources of the following paper:

<i>"Using representer point selection for interpretable drug response prediction 
" Under review. </i>


## Overview of the Model


<p align="center">
<img  src="figure/OpenDrug1.png"> 
</p>

OpenDrug is a two step interpretable deep learning model for drug response prediction. OpenDrug is a kind of post-hoc
 interpretable deep learning model methods which identify important feature and training example by reanalysis model 
 after 
  whole training process.
### Step 1. Train OpenDrug to predict drug response.
For each drug, we trained a multi-layer neural network model to predict the drug response based on the molecular 
features (gene expression and mutation feature) of each cell
 
### Step 2. Interpreting OpenDrug and identify important training example and feature.
For each drug and each testing cell line, the importance of training data and feature can be interpreted by 
reanalysis the neural network.  




For further details, see Online Methods of our paper. 

## Sub-directories
  - [**src**] contains implementation of OpenDrug used for the beer review data. 
  - [**example_data**] contains one example drug data. The drug response data is save as pickle file.
  - [**baseline_methods**] contains the implementation of baseline methods including elastic net, linear model, and random 
  forest used
   in comparison.
  - [**evaluation**] contains the implementation of GCN used in evaluating important training example identified by 
  OpenDrug
  - [**ipynb**] contains the tutorial for processing data

<br>

## Data
  - We constructed dataset based on Genomics of Drug Sensitivity in Cancer (GDSC) project(Iorio et al., 2016), which 
  measured the drug responses elicited by a panel of 265 drugs on 1,001 tumor cell lines, which can be found and 
  downloaded at the GDSC website (https://www.cancerrxgene.org/gdsc1000/GDSC1000_WebResources)
  
   - We selected genes from the “mini cancer genome panel” and further removed gene expression features for which the 
   standard 
   deviations fell into the lowest 20% over all the 
   training samples, and gene mutation features with less than 5 somatic mutations across all the training samples.
   
  - We also provide a process tutorial in ipynb/GDSC_data_preprocessing.ipynb
    
<br>

## Dependency
- python 3.7
- pytorch 1.7.0+cu101
- ogb 1.2.4+cu101
- scikit-learn 0.23.2
- lime 0.2.0.1
- ax-platform 0.1.19


## Code Usage
1. Train OpenDrug to predict drug response. 
    - Benchmark drug response performance for OpenDrug
        ```
        cd src
        python train.py --drug_id <drug_id> --data_dir <data_dir> 
        ```
    - Leave-one-out cross-validation implementation which can be used in further interpreting model.
        ```
        cd src
        python train.py --drug_id <drug_id> --data_dir <data_dir> --model_dir <model_dir>
        ```
    - Baseline method implementations used in our paper.
        ```
        cd baseline_methods
        python run_{elastic_net,linear_model,rf}.py  --drug_id <drug_id> --data_dir <processed data path> --model_dir <path to save model> 
        ```

2. Interpreting OpenDrug and identify important training example and feature.
    ```
    cd src
    python generate_representer_matrix_index.py --drug_id <drug_id> --data_dir <processed data path> --index_list <test_index_list> 
    --model_dir <model path> --output_dir <path to save interprate result> 
    ```
    The key idea of OpenDrug is that it can interpret drug response prediction model in a post-hoc way. For each 
    test tumor sample, this script 
    can be used to identify the most important training sample for the prediction (relate to the manuscript Eq.5) and
     sample-specific feature(gene) importance for the prediction (related to manuscript Eq.6).
    
    Besides precessed data file and model file trained in step 1, the input files also including a test_index_list file which denotes the sample index that need to be interpreted.
    
    
<<<<<<< HEAD
    This script generate two numpy format file, feature importance and sample importance. "\*\_feature_importance.npy" 
    denotes to feature importance interprated by OpenDrug, the matrix weight (W<sub>ij</sub>) represent importance of
     feature j on training data. And "\*\_sample_importance.npy" denotes to  the sample importance interprated by 
=======
    This script generate two numpy format file, feature importance and sample importance. "*\_feature\_importance.npy" 
    denotes to feature importance interprated by OpenDrug, the matrix weight (W<sub>ij</sub>) represent importance of
     feature j on training data. And "*\_sample\_importance.npy" denotes to  the sample importance interprated by 
>>>>>>> 1407548 (update README)
     OpenDrug, 
     the matrix weight W<sub>i</sub> represent importance of sample i.


3. Evaluating important training example and feature

    **Evaluation 1:** OpenDrug perform outperform other methods by using identified important training feature. 
    ```
    cd evaluation
    python Open_Drug_feature_selection.py --drug_id <drug_id> --data_dir <processed data path> --index_list 
    <test_index_list> --model_dir <saved training model> --output_dir <path to save performance result> 
    --representer_value_matrix_dir_prefix <path of representer graph calculated in step 2> 
    ```
    Similarly, other feature interpreting baseline can be evaluated by 
    ```
    cd evaluation
    python random_feature_selection.py --drug_id <drug_id> --data_dir <processed data path> --index_list 
    <test_index_list> --output_dir <path to save performance result> 
    python lime_feature_selection.py --drug_id <drug_id> --data_dir <processed data path> --index_list 
    <test_index_list> --output_dir <path to save performance resault> 
    ```
   The output result is saved in pickle file including predicted drug response and test data index.
   
   
   **Evaluation 2:** Verifying the representers using graph convolutional networks (GCN).
   
   ```
   cd evaluation
   python train_gcn.py --data_path <path save processed data> --type <random/OpenDrug> 
   --representer_value_matrix_dir_prefix < only used for OpenDrug, path of representer graph calculated in 
   step 2>
   ```
    The output result is saved in pickle file including person correlation with different percentage of top feature 
  selected.
  
    ---
    If you have any question, please feel free to contact to me.
Email: sht18@mails.tsinghua.edu.cn, majianzhu@gmail.com

    ---

## License
OpenDrug is licensed under the Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
