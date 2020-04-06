Due to the size of the file, we have compressed and split the training set data file and the model file trained by PhysOrg-RF:   
  - MolGraph_Ar_dev_2.pt  
  - MolGraph_R_dev_2.pt  
  - ReactionDataset_ArR_dev_2.pt   
  - SelectivityDataset_ArR_dev_2.pt
  - models_pkg/Predictor_SOAP_fp@XGB_Reg_20200328_035409.pkl

And you could execute the following command to restore the original file with bash shell command:  
         
    unzip MolGraph_Ar_dev_2.zip    
    unzip MolGraph_R_dev_2.zip    
    cat ReactionDataset_ArR_dev_2.zip.* | unzip      
    cat SelectivityDataset_ArR_dev_2.zip.* | unzip      
    cd models_pkg    
    cat PhyChem@RF_Reg.zip.* | unzip     

Once we get the Pt file, we no longer need the structure file. But other files in **../../raw/dev_2** are necessary.
   
Models, charts and figures generated through the training and testing process are also saved in this folder
