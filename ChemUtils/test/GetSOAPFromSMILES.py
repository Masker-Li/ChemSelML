# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 08:51:21 2020

@author: Licheng
"""
import glob,os,ase,shutil
from openbabel.pybel import (readfile,Outputfile) 
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from dscribe.descriptors import SOAP
from ase.io import read

#%%
# smi to Embed Geometry and MMFF94 Geometry
#### Please be careful to run this cell
def GenerateGeometryByRDKit(smi_file:str,Geometry_folder:str,geom_num=5,Radical=False):
    if not os.path.exists('./Geometry'):
        os.mkdir('./Geometry')
    if not os.path.exists('./Geometry/Ar'):
        os.mkdir('./Geometry/Ar')
    if not os.path.exists('./Geometry/R'):
        os.mkdir('./Geometry/R')
    periodic_table = Chem.GetPeriodicTable()
    with open(smi_file,'r') as fr:
        lines = fr.readlines()
    name_list = [line.strip().split()[0] for line in lines]
    smiles_list = [line.strip().split()[1] for line in lines]
    #embed_folder = Geometry_folder + 'Embed/'
    mmff94_folder = Geometry_folder + 'MMFF94/'
    #pm7_folder = Geometry_folder + 'PM7/'
   
    
    #if not os.path.exists(embed_folder):
    #    os.mkdir(embed_folder)
    if not os.path.exists(mmff94_folder):
        os.mkdir(mmff94_folder)
    #if not os.path.exists(pm7_folder):
    #    os.mkdir(pm7_folder)
        
    for name,smiles in zip(name_list,smiles_list):
        
        
        #geom_embed_sub_folder = embed_folder + name + '/'
        geom_mmff94_sub_folder = mmff94_folder + name + '/'
        #geom_pm7_sub_folder = pm7_folder + name +'/'
        #if not os.path.exists(geom_embed_sub_folder):
        #    os.mkdir(geom_embed_sub_folder)
        if not os.path.exists(geom_mmff94_sub_folder):
            os.mkdir(geom_mmff94_sub_folder)
        #if not os.path.exists(geom_pm7_sub_folder):
        #    os.mkdir(geom_pm7_sub_folder)
            
        for i in range(geom_num):
            conf_string = "%nproc=8\n%mem=6GB\n#p opt freq pm7\n\n Title\n\n"
            tmp_mol = Chem.MolFromSmiles(smiles)
            mol = AllChem.AddHs(tmp_mol)
            converge_flag = AllChem.EmbedMolecule(mol)
            #if converge_flag == 0:
            #    Chem.MolToMolFile(mol,geom_embed_sub_folder+name+'-'+str(i)+'.sdf')
            #else:
            #    print("Something wrong (embed) with file: %s"%name)
            opt_flag = AllChem.MMFFOptimizeMolecule(mol)
            if opt_flag == 0:
                Chem.MolToMolFile(mol,geom_mmff94_sub_folder+name+'-'+str(i)+'.sdf')
            else:
                print("Something wrong (opt) with file: %s"%name)
            if not Radical:
                    
                AllChem.ComputeGasteigerCharges(mol)
                charges = np.array([eval(mol.GetAtomWithIdx(i).GetProp('_GasteigerCharge')) for i in range(mol.GetNumAtoms())])
                charge = charges.sum()
                if charge > 10e-3:
                    charge = 1
                    
                else:
                    charge = 0
                conf_string += str(charge)+" 1\n"
            else:
                conf_string += "0 2 \n"
            atom_type_list = [periodic_table.GetElementSymbol(atom.GetAtomicNum()) for atom in mol.GetAtoms()]
            positions = mol.GetConformer().GetPositions()
            for atom,pos in zip(atom_type_list,positions):
                conf_string += "%2s %15f % 15f %15f \n"%(atom,pos[0],pos[1],pos[2])
            conf_string += '\n\n'
            #with open(geom_pm7_sub_folder+name+'-'+str(i)+'.gjf','w') as fw:
            #    fw.writelines(conf_string)
            
Ar_smi = './smi/Ar.smi'
Ar_Geometry_folder = './Geometry/Ar/'
R_smi = './smi/R.smi'
R_Geometry_folder = './Geometry/R/'

GenerateGeometryByRDKit(Ar_smi,Ar_Geometry_folder,5,False)
GenerateGeometryByRDKit(R_smi,R_Geometry_folder,5,True)

#%%
# Generate SOAP Descriptors
def Log2Atoms(log_file):
    periodic_table = Chem.GetPeriodicTable()
    with open(log_file,'r') as fr:
        lines = fr.readlines()
    coord_start_index_list = []
    
    for i,line in enumerate(lines):
        if 'NAtoms=' in line:
            atom_num = eval(line.split()[1])
        if 'Standard orientation' in line:
            coord_start_index_list.append(i+5)
    coord_string = lines[coord_start_index_list[-1]:coord_start_index_list[-1]+atom_num]
    coord = np.array([[eval(item.strip().split()[3]),eval(item.strip().split()[4]), eval(item.strip().split()[5])] for item in coord_string])
    atom_type = [periodic_table.GetElementSymbol(eval(item.split()[1])) for item in coord_string]
    atom_type_string = ''.join(atom_type)
    atoms = ase.Atoms(symbols=atom_type_string,positions=coord)
    return atoms
def MolFormatConversion(input_file:str,output_file:str,input_format="xyz",output_format="sdf"):
    molecules = readfile(input_format,input_file)
    output_file_writer = Outputfile(output_format,output_file)
    for i,molecule in enumerate(molecules):
        output_file_writer.write(molecule)
    output_file_writer.close()
    print('%d molecules converted'%(i+1))
def Log2sdf(log_file,sdf_file):
    periodic_table = Chem.GetPeriodicTable()
    work_dir = '/'.join(log_file.split('/')[:-1]) + '/'
    name = log_file.split('/')[-1].split('.')[0]
    with open(log_file,'r') as fr:
        lines = fr.readlines()
    coord_start_index_list = []
    
    for i,line in enumerate(lines):
        if 'NAtoms=' in line:
            atom_num = eval(line.split()[1])
        if 'Standard orientation' in line:
            coord_start_index_list.append(i+5)
    coord_string = lines[coord_start_index_list[-1]:coord_start_index_list[-1]+atom_num]
    coord = np.array([[eval(item.strip().split()[3]),eval(item.strip().split()[4]), eval(item.strip().split()[5])] for item in coord_string])
    atom_type = [periodic_table.GetElementSymbol(eval(item.split()[1])) for item in coord_string]
    xyz_string = '%d\n%s\n'%(len(atom_type),name)
    for atom,c in zip(atom_type,coord):
        xyz_string += '%2s %15f %15f %15f\n'%(atom,c[0],c[1],c[2])
    xyz_file = work_dir+name+'.xyz'
    with open(xyz_file,'w') as fw:
        fw.writelines(xyz_string)
    MolFormatConversion(xyz_file,sdf_file,'xyz','sdf')
    os.remove(xyz_file)
    ### Stop here
def Get_Key_atom_num(sdf_file):
    special = {"C0h":[2,3], "D0h":[2,3], "E0h":[2,3], "G0h":[1,6,5], "G3c":[4,5],
               "G3d":[4,5], "H0h":[1] , "I0h":[1], "I1c":[5], "I1d":[5],'HFe0':[1,4],
               "HFe2c":[6],"HFe2d":[6],"HFf0":[1,2],"HFf4c":[4],"HFf4d":[4],
               "G0hH":[1,5,6],"G3cH":[4,5],"G3dH":[4,5],"H0hH":[4],"H0hH-c2":[1],
               "I0hH":[5], "I1cH":[7],"I1cH-c2":[5],"I1cH-c3":[5],
               'I0hH-c2':[1],'I1dH':[7],'I1dH-c2':[5],'I1dH-c3':[5],
               'G3t':[6,7],'G3cHt':[6,7],'G3cHb':[6,7],'G3b':[6,7]
               }
    keys = list(special.keys())
    name = sdf_file.split('/')[-1].split('-')[0]
    if name in keys:
        return list(np.array(special[name])-1)
    else:
        key_atom_nums = []
        mols = Chem.SDMolSupplier(sdf_file,removeHs=False)
        for mol in mols:
            pass
        atom_in_ring = mol.GetRingInfo().AtomRings()[0]          # In the ring
        for index in atom_in_ring:
            atom = mol.GetAtomWithIdx(index)
            symbol = atom.GetSymbol()
            neighbors = atom.GetNeighbors()
            neighbors_symbols = [neighbor.GetSymbol() for neighbor in neighbors]
            if symbol == 'C' and 'H' in neighbors_symbols:       # is Carbon and connect with hydrogen
                key_atom_nums.append(index)
        return key_atom_nums
def Get_Key_atom_num_Radical(sdf_file):
    #name = sdf_file.split('/')[-1].split('.')[0]
    key_atom_nums = []
    mols = Chem.SDMolSupplier(sdf_file,removeHs=False)
    for mol in mols:
        pass
    atoms = mol.GetAtoms()
    for i,atom in enumerate(atoms):
        if atom.GetSymbol() == 'C' and atom.GetTotalValence() == 3:
            key_atom_nums.append(i)
    return key_atom_nums
def SOAP_Definition(species=None):
    if not species:
        species = ["H", "C", "N", "O", "F", "S"]
    rcut = 6.0
    nmax = 8
    lmax = 6
    
    # Setting up the SOAP descriptor
    soap = SOAP(
        species=species,
        periodic=False,
        rcut=rcut,
        nmax=nmax,
        lmax=lmax,)
    return soap

    
def get_SOAP(atoms,soap,key_atom_num=None):
    atom_nums = atoms.get_atomic_numbers()
    atom_nums[atom_nums>16]=1
    atoms.set_atomic_numbers(atom_nums)
    
    mol_soap = soap.create(atoms, positions=[key_atom_num])
    return mol_soap
def GetSOAP(geometry_folder,SOAP_csv_file,EmbedGeometry_folder='',Radical=False,type_='RDKit',geom_num=5,species=None):
    
    # type_ = RDKit or Gaussian
    # geometry_folder : e.g. Geometry/Ar/Embed 
    soap = SOAP_Definition(species=species)
    csv_string = ''
    soap_len = 5292
    title = 'molecule,conformer,atom,'
    soap_title_list = ['SOAP %d'%i for i in range(soap_len)]
    soap_title = ','.join(soap_title_list)
    title += soap_title + '\n'
    csv_string += title
    if type_=='RDKit':
        sdf_folders = glob.glob(geometry_folder+'*/')
        sdf_folders.sort()
        for sdf_folder in sdf_folders:
            molecule = sdf_folder.split('/')[-2]
            sdf_files = glob.glob(sdf_folder+'*.sdf')
            sdf_files.sort()
            for j in range(geom_num):
                sdf_file = sdf_files[j]
                atoms = read(sdf_file,format="sdf")
                if not Radical:
                    key_atom_nums = Get_Key_atom_num(sdf_file)
                    for key_atom_num in key_atom_nums:
                        soap_list = list(map(str,get_SOAP(atoms,soap,key_atom_num)[0]))
                        soap_str = ','.join(soap_list)
                        csv_string += '%s,%d,%d,'%(molecule,j,key_atom_num) + soap_str + '\n'
                        
                
                else:
                    key_atom_nums = Get_Key_atom_num_Radical(sdf_file)
                    for key_atom_num in key_atom_nums:
                        soap_list = list(map(str,get_SOAP(atoms,soap,key_atom_num)[0]))
                        soap_str = ','.join(soap_list)
                        csv_string += '%s,%d,%d,'%(molecule,j,key_atom_num) + soap_str + '\n'
        
    elif type_ == 'Gaussian':
        log_folders = glob.glob(geometry_folder+'*/')
        log_folders.sort()
        sdf_std_folders = glob.glob(EmbedGeometry_folder+'*/')
        sdf_std_folders.sort()
        for log_folder,sdf_std_folder in zip(log_folders,sdf_std_folders):
            log_molecule = log_folder.split('/')[-2]
            sdf_std_molecule = sdf_std_folder.split('/')[-2]
            if log_molecule != sdf_std_molecule:
                print('There is something wrong with file, log : %s, sdf: %s'%(log_molecule,sdf_std_molecule))
                continue
            log_files = glob.glob(log_folder+'*.log')
            log_files.sort()
            sdf_std_file = glob.glob(sdf_std_folder+'*.sdf')[0]
            for j in range(geom_num):
                log_file = log_files[j]
                #sdf_file = log_file.split('.')[0] + '.log'
                #Log2sdf(log_file,sdf_file)
                atoms = Log2Atoms(log_file)
                if not Radical:
                    key_atom_nums = Get_Key_atom_num(sdf_std_file)
                    for key_atom_num in key_atom_nums:
                        soap_list = list(map(str,get_SOAP(atoms,soap,key_atom_num)[0]))
                        soap_str = ','.join(soap_list)
                        csv_string += '%s,%d,%d,'%(log_molecule,j,key_atom_num) + soap_str + '\n'
                else:
                    key_atom_nums = Get_Key_atom_num_Radical(sdf_std_file)
                    for key_atom_num in key_atom_nums:
                        soap_list = list(map(str,get_SOAP(atoms,soap,key_atom_num)[0]))
                        soap_str = ','.join(soap_list)
                        csv_string += '%s,%d,%d,'%(log_molecule,j,key_atom_num) + soap_str + '\n'
    with open(SOAP_csv_file,'w') as fw:
        fw.writelines(csv_string)
#geometry_folder_Ar_embed = './Geometry/Ar/Embed/'
#SOAP_csv_file_Ar_embed = './SOAP/Ar_Embed_SOAP.csv'
#GetSOAP(geometry_folder_Ar_embed, SOAP_csv_file_Ar_embed,EmbedGeometry_folder='',Radical=False,type_='RDKit',geom_num=5)
   
#geometry_folder_R_embed = './Geometry/R/Embed/'
#SOAP_csv_file_R_embed = './SOAP/R_Embed_SOAP.csv'
#GetSOAP(geometry_folder_R_embed, #SOAP_csv_file_R_embed,EmbedGeometry_folder='',Radical=True,type_='RDKit',geom_num=5)


#geometry_folder_Ar_mmff94 = './Geometry/Ar/MMFF94/'
#SOAP_csv_file_Ar_mmff94 = './SOAP/Ar_MMFF94_SOAP.csv'
#GetSOAP(geometry_folder_Ar_mmff94, SOAP_csv_file_Ar_mmff94,EmbedGeometry_folder='',Radical=False,type_='RDKit',geom_num=5)
   
#geometry_folder_R_mmff94 = './Geometry/R/MMFF94/'
#SOAP_csv_file_R_mmff94 = './SOAP/R_MMFF94_SOAP.csv'
#GetSOAP(geometry_folder_R_mmff94, SOAP_csv_file_R_mmff94,EmbedGeometry_folder='',Radical=True,type_='RDKit',geom_num=5)


#geometry_folder_Ar_PM7 = './Geometry/Ar/PM7/'
#geometry_folder_Ar_embed = './Geometry/Ar/Embed/'
#SOAP_csv_file_Ar_PM7 = './SOAP/Ar_PM7_SOAP.csv'
#GetSOAP(geometry_folder_Ar_PM7, SOAP_csv_file_Ar_PM7,EmbedGeometry_folder=geometry_folder_Ar_embed,Radical=False,type_='Gaussian',geom_num=5)
   
#geometry_folder_R_PM7 = './Geometry/R/PM7/'
#geometry_folder_R_embed = './Geometry/R/Embed/'
#SOAP_csv_file_R_PM7 = './SOAP/R_PM7_SOAP.csv'
#GetSOAP(geometry_folder_R_PM7, SOAP_csv_file_R_PM7,EmbedGeometry_folder=geometry_folder_R_embed,Radical=True,type_='Gaussian',geom_num=5)

#%%
def GetSOAPwithFP(geometry_folder,SOAP_csv_file,EmbedGeometry_folder='',Radical=False,type_='RDKit',geom_num=5,species=None):
    if not os.path.exists('./SOAP'):
        os.mkdir('./SOAP')
    # type_ = RDKit or Gaussian
    # geometry_folder : e.g. Geometry/Ar/Embed 
    soap = SOAP_Definition(species=species)
    csv_string = ''
    soap_len = 5292
    MACCS_fp_len = 167
    Morgan_fp_len = 512
    title = 'molecule,conformer,atom,'
    soap_title_list = ['SOAP %d'%i for i in range(soap_len)]
    soap_title = ','.join(soap_title_list)
    MACCS_fp_tile_list = ['MACCS FP %d'%i for i in range(MACCS_fp_len)]
    MACCS_fp_title = ','.join(MACCS_fp_tile_list)
    Morgan_fp_title_list = ['Morgan FP %d'%i for i in range(Morgan_fp_len)]
    Morgan_fp_title = ','.join(Morgan_fp_title_list)
    title += MACCS_fp_title + ',' + Morgan_fp_title + ',' + soap_title + '\n'
    csv_string += title
    if type_=='RDKit':
        sdf_folders = glob.glob(geometry_folder+'*/')
        sdf_folders.sort()
        for sdf_folder in sdf_folders:
            molecule = sdf_folder.split('/')[-2]
            sdf_files = glob.glob(sdf_folder+'*.sdf')
            sdf_files.sort()
            for j in range(geom_num):
                sdf_file = sdf_files[j]
                atoms = read(sdf_file,format="sdf")
                mol = Chem.MolFromMolFile(sdf_file)
                MACCSfp = AllChem.GetMACCSKeysFingerprint(mol)
                MACCSfp = [int(x) for x in MACCSfp.ToBitString()]
                Morganfp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=512)
                Morganfp = [int(x) for x in Morganfp.ToBitString()]
                mergefp = MACCSfp + Morganfp
                mergefp_str = ','.join(list(map(str,mergefp))) + ','
                #csv_string += mergefp_str
                if not Radical:
                    key_atom_nums = Get_Key_atom_num(sdf_file)
                    for key_atom_num in key_atom_nums:
                        soap_list = list(map(str,get_SOAP(atoms,soap,key_atom_num)[0]))
                        soap_str = ','.join(soap_list)
                        csv_string += '%s,%d,%d,'%(molecule,j,key_atom_num) + mergefp_str + soap_str + '\n'
                        
                
                else:
                    key_atom_nums = Get_Key_atom_num_Radical(sdf_file)
                    for key_atom_num in key_atom_nums:
                        soap_list = list(map(str,get_SOAP(atoms,soap,key_atom_num)[0]))
                        soap_str = ','.join(soap_list)
                        csv_string += '%s,%d,%d,'%(molecule,j,key_atom_num) + mergefp_str +soap_str + '\n'
        
    elif type_ == 'Gaussian':
        log_folders = glob.glob(geometry_folder+'*/')
        log_folders.sort()
        sdf_std_folders = glob.glob(EmbedGeometry_folder+'*/')
        sdf_std_folders.sort()
        for log_folder,sdf_std_folder in zip(log_folders,sdf_std_folders):
            log_molecule = log_folder.split('/')[-2]
            sdf_std_molecule = sdf_std_folder.split('/')[-2]
            if log_molecule != sdf_std_molecule:
                print('There is something wrong with file, log : %s, sdf: %s'%(log_molecule,sdf_std_molecule))
                continue
            log_files = glob.glob(log_folder+'*.log')
            log_files.sort()
            sdf_std_file = glob.glob(sdf_std_folder+'*.sdf')[0]
            mol = Chem.MolFromMolFile(sdf_std_file)
            MACCSfp = AllChem.GetMACCSKeysFingerprint(mol)
            MACCSfp = [int(x) for x in MACCSfp.ToBitString()]
            Morganfp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=512)
            Morganfp = [int(x) for x in Morganfp.ToBitString()]
            mergefp = MACCSfp + Morganfp
            mergefp_str = ','.join(list(map(str,mergefp))) + ','
            #csv_string += mergefp_str
            for j in range(geom_num):
                log_file = log_files[j]
                #sdf_file = log_file.split('.')[0] + '.log'
                #Log2sdf(log_file,sdf_file)
                atoms = Log2Atoms(log_file)
                if not Radical:
                    key_atom_nums = Get_Key_atom_num(sdf_std_file)
                    for key_atom_num in key_atom_nums:
                        soap_list = list(map(str,get_SOAP(atoms,soap,key_atom_num)[0]))
                        soap_str = ','.join(soap_list)
                        csv_string += '%s,%d,%d,'%(log_molecule,j,key_atom_num) + mergefp_str + soap_str + '\n'
                else:
                    key_atom_nums = Get_Key_atom_num_Radical(sdf_std_file)
                    for key_atom_num in key_atom_nums:
                        soap_list = list(map(str,get_SOAP(atoms,soap,key_atom_num)[0]))
                        soap_str = ','.join(soap_list)
                        csv_string += '%s,%d,%d,'%(log_molecule,j,key_atom_num) + mergefp_str + soap_str + '\n'
    with open(SOAP_csv_file,'w') as fw:
        fw.writelines(csv_string)

#geometry_folder_Ar_embed = './Geometry/Ar/Embed/'
#SOAP_csv_file_Ar_embed = './SOAP/Ar_Embed_SOAP_fp.csv'
#GetSOAPwithFP(geometry_folder_Ar_embed, SOAP_csv_file_Ar_embed,EmbedGeometry_folder='',Radical=False,type_='RDKit',geom_num=5)
   
#geometry_folder_R_embed = './Geometry/R/Embed/'
#SOAP_csv_file_R_embed = './SOAP/R_Embed_SOAP_fp.csv'
#GetSOAPwithFP(geometry_folder_R_embed, SOAP_csv_file_R_embed,EmbedGeometry_folder='',Radical=True,type_='RDKit',geom_num=5)


geometry_folder_Ar_mmff94 = './Geometry/Ar/MMFF94/'
SOAP_csv_file_Ar_mmff94 = './SOAP/Ar_MMFF94_SOAP_fp.csv'
GetSOAPwithFP(geometry_folder_Ar_mmff94, SOAP_csv_file_Ar_mmff94,EmbedGeometry_folder='',Radical=False,type_='RDKit',geom_num=5)
   
geometry_folder_R_mmff94 = './Geometry/R/MMFF94/'
SOAP_csv_file_R_mmff94 = './SOAP/R_MMFF94_SOAP_fp.csv'
GetSOAPwithFP(geometry_folder_R_mmff94, SOAP_csv_file_R_mmff94,EmbedGeometry_folder='',Radical=True,type_='RDKit',geom_num=5)


#geometry_folder_Ar_PM7 = './Geometry/Ar/PM7/'
#geometry_folder_Ar_embed = './Geometry/Ar/Embed/'
#SOAP_csv_file_Ar_PM7 = './SOAP/Ar_PM7_SOAP_fp.csv'
#GetSOAPwithFP(geometry_folder_Ar_PM7, SOAP_csv_file_Ar_PM7,EmbedGeometry_folder=geometry_folder_Ar_embed,Radical=False,type_='Gaussian',geom_num=5)
   
#geometry_folder_R_PM7 = './Geometry/R/PM7/'
#geometry_folder_R_embed = './Geometry/R/Embed/'
#SOAP_csv_file_R_PM7 = './SOAP/R_PM7_SOAP_fp.csv'
#GetSOAPwithFP(geometry_folder_R_PM7, SOAP_csv_file_R_PM7,EmbedGeometry_folder=geometry_folder_R_embed,Radical=True,type_='Gaussian',geom_num=5)


