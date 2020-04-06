ChemSelML
=====        
### A trivial demo of chemical regioselectivity prediction via machine learning        
<br>     
     
<!-- TOC START min:1 max:3 link:true asterisk:false update:true -->
- [Project Requirements](#project-requirements)
- [Project Features and User Guide](#project-features-and-user-guide)
  - [Part 1](#part-1)
  - [Part 2](#part-2)
  - [Part 3.](#part-3)
  - [part 4.](#part-4)
<!-- TOC END -->

# Project Requirements
**We generated the relevant packages that the project relies on as requirements.txt files**
These packages can be installed directly in batches using CONDA:   
    `conda install --yes --file requirements.txt`

    ase=3.18.0=pypi_0
    bidict=0.18.3=pypi_0
    dscribe=0.2.9=pypi_0
    joblib=0.13.2=py36_0
    matplotlib=3.1.2=pypi_0
    molml=0.9.0=pypi_0
    networkx=2.3=py_0
    numpy=1.17.4=pypi_0
    openbabel=3.0.0=py36h1360c68_0
    pandas=0.25.0=py36hb3f55d8_0
    pathlib=1.0.1=pypi_0
    pickleshare=0.7.5=py36_0
    pytorch=1.1.0=py3.6_cuda10.0.130_cudnn7.5.1_0
    rdkit=2019.03.2=py36hb31dc5d_1
    scikit-learn=0.22=pypi_0
    scipy=1.2.1=pypi_0
    seaborn=0.9.0=pypi_0
    torch-cluster=1.4.2=pypi_0
    torch-geometric=1.3.0=pypi_0
    torch-scatter=1.3.1=pypi_0
    torch-sparse=0.4.0=pypi_0
    xgboost=0.90=pypi_0

  - It is recommended to execute this project in a linux environment

# Project Features and User Guide
## Part 1
**Optimization of the molecular structure of reaction precursors (heterocyclic substrates and radicals) and calculation of their quantitative property data**

  1. Use quantum chemical software, such as Gaussian 09 or Gaussian 16, to obtain optimized structures at the level of ***B3LYP/6-311+G(2d, p).***
  2. Rearrange the output file of the optimization calculation into a single point calculation input file and place it in a **directory A (for example: ./Example/part_1/Sub/Ar)**. ***Calculation method, functional and basis set*** can be written arbitrarily, and subsequent scripts will automatically correct them.  
    - Tips:  
      - In order to use the scripts for the property calculations, it is recommended that the input files are named in the following format.   
            
            For heterocycles:     
                Ar_n-loc_n-c1-1sp.gjf (such as X4t-1-c1-1sp.gjf)  
                (Ar_n means the heterocycle's alias,
                loc_n means one of the atomic order number of the reaction site in heterocycle,
                c1 means the first conformation，
                and the sp suffix indicates that it is a single point calculation file.)
            For Radicals:       
                R_n-c1-1sp.gjf (such as CF3-c1-1sp.gjf)
      - For easy extraction of local descriptors, please add the atomic order number of the reaction site to the title line of the heterocycle's input file.     
                     
            For example:    
            in the fifth line of **./Example/part_1/Sub/Ar/X4t-1-c1-1sp.gjf**,    
            1 and 3 means the first and third atomic order number of the reaction    
            site in this heterocycle.   
               
          > %nprocshared=28  
          > %mem=56GB  
          > #p aug-cc-pvtz m062x g09def  
          >   
          > X4t-1-c1-1stdsp **1 3**  
          >   
          > 0 1  
          > **C &nbsp; &nbsp;  &nbsp;  &nbsp;  -1.42417300&nbsp; &nbsp;  &nbsp;  &nbsp;   2.11888500 &nbsp; &nbsp;  &nbsp;  &nbsp;   0.00012700**    
          >  N &nbsp; &nbsp;  &nbsp;  &nbsp;  -0.21250800 &nbsp; &nbsp;  &nbsp;  &nbsp;   2.61217900 &nbsp; &nbsp;  &nbsp;  &nbsp;  -0.00003700  
          >  **C &nbsp; &nbsp;  &nbsp;  &nbsp;  0.80986200 &nbsp; &nbsp;  &nbsp;  &nbsp;   1.72740500 &nbsp; &nbsp;  &nbsp;  &nbsp;   -0.00011600**    
          >  C &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;&nbsp;  0.71915500 &nbsp; &nbsp;  &nbsp;  &nbsp;   0.34596500 &nbsp; &nbsp;  &nbsp;  &nbsp; &nbsp; 0.00003400  
          >  C &nbsp; &nbsp;  &nbsp;  &nbsp; -0.61252800&nbsp; &nbsp;  &nbsp;  &nbsp;  -0.19945400&nbsp; &nbsp;  &nbsp;  &nbsp; &nbsp;&nbsp;  0.00007300  
          > ...
          >   

  3. Copy the **Descriptors_input_trans_new.py** and **Input_Make_NICS.py** scripts from the **./ChemUtils/test** folder to that **directory A** (for example: **./Example/part_1/Sub/Ar**) and run them separately to get new Gaussian input files of a quantitative properties, which are placed in the **Desc_gas_sp** and **NICS** folders, respectively.In particular, for free radicals, there is no need to run the **Input_Make_NICS.py** script.
  4. Submit files from these two folders to Gaussian Software for calculation and place the resulting output files in the same directory.
  5. Copy **get_PhysOrg.py** to the **Desc_gas_sp and NICS's parent directory A** (for example: **./Example/part_1/Sub/Ar**). Edit **get_PhysOrg.py** and **change the folder absolute path of the project to the  ChemUtils's parent folder (for example the folder absolute path of this tutorial file) in the third line**, then save and close this file. Run it and therein will automatically generate the PhysOrg descriptor in the current directory in the **phychem.csv** file (**for example: ./Example/part_1/Sub/Ar/Ar_phychem.csv**).
         
   - The purpose of rewriting the third line is to add the absolute path of the directory where ChemUtils is located to the python package search path   
        
  > import os  
  > import sys  
  > sys.path.append(**"/mnt/G16/Scripts"**)  
  >  
  > from ChemUtils.bin.PhysOrg import get_Pre_EQBV, get_Pre_PhysOrg  
  > ...  
  >    


## Part 2  
**Get SOAP/FP descriptors based on different structures from SMILES**  

The operation in this part is to prepare for the SOAP/FP-XGB model based on diversified structure. You can skip this section if you don't need it.

  1. Generate the SMILES of the structures, and put it in **smi** (**for example: ./Example/part_2/smi**) folder, named Ar.smi and R.smi separately.  
  2. Copy the **GetSOAPFromSMILES.py** script from the **./ChemUtils/test** folder to **smi's parent folder (for example: ./Example/part_2)** and run it to Generate 5 different 3D geometries by RDKit based on SMILES at MMFF94 level. The generated geometries will placed in **Geometry (for example: ./Example/part_2/Geometry)** folder and the descriprors of SOAP/FP will placed in **SOAP (for example: ./Example/part_2/SOAP)** folder.

  All the required code has been provided in **ChemUtils/test/GetSOAPFromSMILES.py**   
  We also provide an example of this in the **./Example/part_2** folder.   


## Part 3.   
**Pre-test/pre-training preparation.**

  1. For test task, create a new folder named start with **test_** such as **test_sub in DataSet/raw** folder to put all the required files in it. **test_** is used to indicate that the data in this folder is used for testing and suffix of **sub** is used to distinguish between different test sets. In the next steps, you also need creat some necessary folders as needed.   
  2. The structure files related to aromatic heterocyclic precursors should be placed in **test_sub/Ar/gjfs_and_logs** and the structure files related to radical precursors should be placed in **test_sub/R/gjfs_and_logs**. These structure files are used to generate a range of other types of descriptors, such as ASCF, SOAP, Fingerprint, etc. Those descriptors could be used in
  Those structure files could be available from the **Desc_gas_sp folder in Part 1** with suffix of **sp.gjf** and **sp.log**.
  3. PhysOrg descriptors file (Ar_phychem.csv and R_phychem.csv) should alse be placed in corresponding folder.(**test_sub/Ar and test_sub/R**)
  4. We also need a label file **TestSet_Label.csv** which should be placed in **test_sub** to show the data set which chemical reaction combination information is included. The label file needs to be organized by the operator according to his needs, and it mainly records the transition state energy barrier information for training or testing, including heterocyclic aromatic aliases **Ar_n**, the atomic order number of the reaction site in heterocycle **loc_n**, radical aliases **Radical_n** and transition state energy barrier **DG_TS** information. For data that only needs to be predicted, **DG_TS** can be filled with 0.0.  
      - The schema of the entire sub folder is as follows:  
                         
             -- DataSet/raw/test_sub
                |-- Ar  
                |  |-- gjfs_and_logs  
                |  |   |-- Ar1.gjf
                |  |   |-- Ar1.log  
                |  |-- Ar_phychem.csv  
                |-- R  
                |  |-- gjfs_and_logs  
                |  |   |-- R1.gjf
                |  |   |-- R1.log  
                |  |-- R_phychem.csv
                |-- TestSet_Label.csv
      - We have shown the location and contents of relevant documents in **./DataSet/raw/test_sub** folder
      - It is worth noting that the information such as **Ar_n, loc_n and Radical_n** in the label file needs to be recorded in the precursor folders Ar and R as described in the previous steps.

  5. Call the ***ReactionDataset*** and ***SelectivityDataset*** classes in ***ChemSelML.bin.ChemSelectivityDataset*** in the project, it will integrate all information in **./DataSet/raw/test_sub** into one reaction database file **ReactionDataset_ArR_test_sub.pt** and chemical selectivity database file **SelectivityDataset_ArR_test_sub.pt**, which includes various feature categories and target attributes needed for model training or testing. The generated file is located in a folder with the same name as **test_sub** under the **./Dataset/processed** folder. See the example to generate this two file in the **./Example/part_4/PhysOrg-RF.ipynb** file for details. Once these PT files are generated, they can be called multiple times.
        
          import numpy as np
          import pandas as pd
          import torch

          import os
          import sys
          # Add the absolute path of the directory where ChemSelML is located to the python package search path
          sys.path.append("/PyScripts/PyTorch.dir/Radical")

          from ChemSelML.bin.ChemSelectivityDataset import ReactionDataset, SelectivityDataset

          # '/PyScripts/PyTorch.dir/Radical/DataSet' corresponds to the "../DataSet" directory in this project.
          # mode corresponds to the folder name in "../DataSet/raw"
          dev2_ArR_dataset = ReactionDataset(root='/PyScripts/PyTorch.dir/Radical/DataSet', mode='dev_2')
          print(dev2_ArR_dataset,'\n')
          print(dev2_ArR_dataset.data,'\n')

          dev2_ArR_DDG_dataset = SelectivityDataset(root='/PyScripts/PyTorch.dir/Radical/DataSet', mode='dev_2')
          print(dev2_ArR_DDG_dataset,'\n')
          print(dev2_ArR_DDG_dataset.data,'\n')  
          
  6. For training task, the created folder names should start with **dev_** and followed by your own suffix identifier, such as **dev_2** and label file should be named with **TrainSet.csv**. And the **test_sub** in the corresponding filename and directory path described in the previous steps will change to **dec_2**.   

      - We provide the complete training set with the required files in the **./Dataset/processed/dev_2** folder and **./Dataset/raw/dev_2** folder      

## part 4.
**Model selection, training and testing**

  1. This project provides a model screening module ***ChemSelML.training.benchmark*** and a predictor training module ***ChemSelML.training.ChemSelPredictor.***
  We show in detail the use of the model screening and predictor training functions in the **./Example/part_4** folder.
  2. For model selection, we've put the relevant code in **./Example/part_4/Benchmark.ipynb**.We have selected different regressors or classifiers according to the requirements to conduct 5-fold cross-validation tests on various combinations of features to compare the performance of each model.
  3. And for model training and test case, we show in detail the training, saving, retuning, and prediction of the entire functional block of the **PhysOrg-RF** and **SOAP/FP(MMFF84)-XGB** models. It is easy for everyone to understand and use. We've put the relevant code in **./Example/part_4/PhysOrg-RF.ipynb** and **./Example/part_4/SOAP/FP(MMFF84)-XGB.ipynb**.   
        - PS.    
                
          We provide an output sample, which is placed in the example folder
          csv output file description:   
              \*_DDG_Pred_ArR_site_sort_\*.csv: the transformation result of the predicted energy barrier difference of the model into the reference energy barrier result of each site.   
              \*_DDG_Pred_site_vs_site_\*.csv: the predicted energy barrier difference of the model   

