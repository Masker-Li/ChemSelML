SMILES string is saved in .smi files which contains lines like ；'C0h c1ccoc1' at smi folder. 

Calling "GenerateGeometryByRDKit" function will generate 3D geometries by RDKit based on .smi files. The geometry file will be saved in sub-folders in "Geometry" folder.

PM7 folder will save the Gaussian input files which be need calculated, and the output files should be saved in same folders.

After preparing all geometry files, we need call "GetSOAP" function or "GetSOAPwithFP" to generate SOAP(or SOAP with molecular fingerprint) describes. Calling "Get_Key_atom_num" function will generate key positions infomation that needed when generate SOAP. "Get_Key_atom_num" function judge key positions at a molecules through bond connection information, if there is geometric partly in a molecule, we need judge key positions information by ourselves and add them to dictionary variable "special" in this function.

