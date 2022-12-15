## Reaction data
### 3D structure files
The **xyz** format structure files can be downloaded from this link:  
http://spmsgen.net:8000/download/Radical_C-H_Functionalization_of_Heterocycles_TrainSet_xyz.zip

### SMILES data
We collated the training set with the reaction data represented by smiles and placed it in **Canonicalized_SMILES_Reactions_input_data.csv**    
Transition state energy barriers are stored under the **Output** column or the **DG_TS** column. They are the same.   
   
Notes:    
The means of **Product_name**: Ar{Num} -> Arene_{Num}; R{Num} -> Radical_{Num}. The columns after **Divider** are raw information that can be used to assist in determining the similarity of reactants and products. The SMILES before the **Divider** column all have been canonicalized.      
The means of **React_sites_Ar_R**: for example: ((7,), (1,)): The first element in the tuple **(7,)** indicates the reactive site on the aromatic ring, which means only one reactive site here; similarly, the second element **(1,)** indicates the reactive site on the radical. The site correspond to the Idx in the **rdkitmol** obtained by rdkit from **Arene_smi** and **Radical_smi**, respectively.
The means of **Prod_sites**: for example: ((5,), (1,)): The first element in the tuple **(5,)** indicates the reactive site on the product correspond to the reaction site **(7,)** of aromatic ring, which means only one reactive site here; similarly, the second element **(1,)** indicates the reactive site on the product correspond to the reaction site **(1,)** of radical. The site correspond to the Idx in the **rdkitmol** obtained by rdkit from **Product_smi**.
