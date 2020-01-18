import numpy as np
import pandas as pd
from bidict import bidict
from rdkit import Chem
import dscribe as ds
from dscribe.descriptors import ACSF
from dscribe.descriptors import CoulombMatrix
from dscribe.descriptors import SOAP

def ascf_Definition(species=None):
    if not species:
        species = ["H", "C", "N", "O", "F", "S"]
    rcut=10
    #G2 - eta/Rs couples:
    g2_params = [[1, 2], [0.1, 2], [0.01, 2],
                 [1, 4], [0.1, 4], [0.01, 4],
                 [1, 6], [0.1, 6], [0.01, 6]] 
    #G4 - eta/ksi/lambda triplets:
    g4_params = [[1, 4,  1], [0.1, 4,  1], [0.01, 4,  1], 
                 [1, 4, -1], [0.1, 4, -1], [0.01, 4, -1]]
    g3_params = None
    g5_params = None
    acsf = ACSF(
                species=species,
                rcut=rcut,
                g2_params=g2_params,
                g3_params=g3_params,
                g4_params=g4_params,
                g5_params=g5_params,
                sparse=False
                )
    return acsf

def sparse_sf_2_dense_sf124(sparse_sf,g2_para_num,g4_para_num,atom_type_num):
    # sum G1
    dense_sf = np.zeros([1+g2_para_num+g4_para_num])
    for i in range(atom_type_num):
        #print((1+g2_para_num)*i)
        dense_sf[0]+= sparse_sf[(1+g2_para_num)*i]
        #print((1+g2_para_num)*i)
    # sum G2
    for i in range(g2_para_num):
        #print(i)
        for j in range(atom_type_num):
            #print((g2_para_num+1)*j+(i+1))
            dense_sf[i+1] += sparse_sf[(g2_para_num+1)*j+(i+1)]
    # sum G4
    for i in range(g4_para_num):
        #print(i)
        for j in range(round(atom_type_num*(atom_type_num+1)/2)):
            #print((1+g2_para_num)*atom_type_num+j*g4_para_num+i)
            dense_sf[1+g2_para_num+i] += sparse_sf[(1+g2_para_num)*atom_type_num+j*g4_para_num+i]
    return dense_sf   #[G1,G2_param1,G2_param2,...,G4_param1,G4_param_2]    

species = ["H", "C", "N", "O", "F", "S"]
acsf = ascf_Definition(species=species)

def get_acsf_features(atoms, key_atom_num=None):
    atom_nums = atoms.get_atomic_numbers()
    atom_nums[atom_nums>16]=1
    atoms.set_atomic_numbers(atom_nums)
    
    #G2 - eta/Rs couples:
    g2_params = [[1, 2], [0.1, 2], [0.01, 2],
                 [1, 4], [0.1, 4], [0.01, 4],
                 [1, 6], [0.1, 6], [0.01, 6]] 
    #G4 - eta/ksi/lambda triplets:
    g4_params = [[1, 4,  1], [0.1, 4,  1], [0.01, 4,  1], 
                 [1, 4, -1], [0.1, 4, -1], [0.01, 4, -1]]
    g3_params = None
    g5_params = None
    
    acsf_features = acsf.create(atoms, positions=key_atom_num, n_jobs=1) 
    # structure of return is [[#acsf features] for each position in molecule_system]
    ascf_features = np.array([sparse_sf_2_dense_sf124(acsf_feature,len(g2_params),len(g4_params),len(species)) 
                      for acsf_feature in acsf_features],dtype=np.float32)
    return ascf_features

def cm_Definition(n_atoms_max):
    cm = CoulombMatrix(
        n_atoms_max=n_atoms_max,
        flatten=False,
        permutation="sorted_l2")
    return cm

cm = cm_Definition(n_atoms_max=29)
    
def get_CoulombMatrix(atoms):
    mol_cm = cm.create(atoms)
    return mol_cm

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

soap = SOAP_Definition(species=species)
    
def get_SOAP(atoms, key_atom_num=None):
    atom_nums = atoms.get_atomic_numbers()
    atom_nums[atom_nums>16]=1
    atoms.set_atomic_numbers(atom_nums)
    
    mol_soap = soap.create(atoms, positions=key_atom_num)
    return mol_soap

class PhyChem:
    def __init__(self, mode, suffix):
        assert mode in ['Ar', 'R'], "mode should be Ar/R"
        root = r'/PyScripts/PyTorch.dir/Radical/DataSet/raw'
        if mode == 'Ar':
            self.csv_f = r'%s/%s/%s'%(root, suffix, 'Ar/Ar_phychem.csv')
            self.total_PhyChem_labels = ['Ar_loc_idx', 'E_HOMO', 'E_LUMO', 'm', 'h',
                        's', 'w', 'NICS_0', 'NICS_1', 'NICS_0_ZZ', 'NICS_1_ZZ']
            self.local_PhyChem_labels = ['Ar_loc_idx', 'QN', 'f1', 'f0', 'f_1', 'Df', 'local_S_f0',
                        'maxC-C', 'C-H', 'midC-C', 'aveB', 'BV_3A', 'BV_4A', 'BV_5A']
        else:
            self.csv_f = r'%s/%s/%s'%(root, suffix, 'R/R_phychem.csv')
            self.total_PhyChem_labels = ['R_idx', 'R_E_SOMO', 'R_IE', 'R_EA', 'R_c(-m)', 'R_S']
            self.local_PhyChem_labels = ['R_idx', 'R_QN', 'R_f0', 'R_local_S_f0', 
                        'R_maxB', 'R_minB', 'R_aveB', 'R_BV_3A', 'R_BV_4A', 'R_BV_5A']
        self.total_PhyChem = pd.read_csv(self.csv_f, index_col=0, usecols=self.total_PhyChem_labels)
        self.local_PhyChem = pd.read_csv(self.csv_f, index_col=0, usecols=self.local_PhyChem_labels)
    
    def get_total_PhyChem(self, idx):
        '''
        idx: int or list of int. used for pandas.DataFrame.loc function
        '''
        _total = self.total_PhyChem.loc[idx,:]
        return _total
    
    def get_local_PhyChem(self, idx):
        '''
        idx: int or list of int. used for pandas.DataFrame.loc function
        '''
        _local = self.local_PhyChem.loc[idx,:]
        return _local
    
def PhyChem_transform(feature_names=None, feature_alias=None):
    assert None in [feature_names, feature_alias], 'Pass in at most one parameter：feature_name or feature_alias'
    assert type(feature_names)==list if feature_names else type(feature_alias)==list
    name_dict = {
        'E1' : 'E_HOMO',  'E2': 'E_LUMO', 'E3': 'm', 'E4' : 'h', 'E5': 's', 'E6' : 'w',
        'NICS1' : 'NICS_0', 'NICS2' : 'NICS_1', 'NICS3' : 'NICS_0_ZZ', 'NICS4' : 'NICS_1_ZZ',
        'Q1' : 'QN', 'Q2' : 'f1', 'Q3' : 'f0', 'Q4': 'f_1', 'Q5' : 'Df', 'Q6' : 'local_S_f0',  
        'B1' : 'maxC-C', 'B2': 'C-H', 'B3' : 'midC-C', 'B4' : 'aveB', 'V1' : 'BV_3A',    
        'V2' : 'BV_4A', 'V3': 'BV_5A', 'RE7': 'R_E_SOMO', 'RE8': 'R_IE', 'RE9': 'R_EA',
        'RE10'  : 'R_c(-m)', 'RE11'  : 'R_S', 'RQ7': 'R_QN', 'RQ8': 'R_f0', 'RQ9': 'R_local_S_f0', 
        'RB5': 'R_maxB', 'RB6': 'R_minB', 'RB7': 'R_aveB', 'RV4': 'R_BV_3A', 'RV5': 'R_BV_4A', 'RV6': 'R_BV_5A' }
    bi_name_dict = bidict(name_dict)
    name_dict_inverse = bi_name_dict.inverse
    if feature_names:
        return [name_dict_inverse[x] for x in feature_names]
    elif feature_alias:
        return [name_dict[x] for x in feature_alias]

def get_RDKit(mol):      
    def get_BondNum(mol):
        neighbor_num = []
        for atom in mol.GetAtoms():
            neighbor_num.append(len(atom.GetNeighbors()))
        bondtype_plus = {Chem.rdchem.BondType.SINGLE:0., Chem.rdchem.BondType.DOUBLE:1. , 
                         Chem.rdchem.BondType.TRIPLE:2., Chem.rdchem.BondType.AROMATIC:0.5}
        for i in range(mol.GetNumAtoms()):
            for j in range(mol.GetNumAtoms()):
                e_ij = mol.GetBondBetweenAtoms(i, j)
                if e_ij is not None:
                    bondtype = e_ij.GetBondType()
                    bond_list = [i,j,bondtype]
                    neighbor_num[i] += bondtype_plus[bondtype]
        return neighbor_num
    
    def get_inRingNum(mol):
        RingNumList = [0 for i in range(mol.GetNumAtoms())]
        ring = mol.GetRingInfo()
        ring_atom_list = ring.AtomRings()
        for ring_atom in ring_atom_list:
            for atom_index_on_ring in ring_atom:
                RingNumList[atom_index_on_ring] += 1
        return RingNumList
    
    def get_SpiroEndo(mol):
        SpiroList = [0 for i in range(mol.GetNumAtoms())]
        EndoList = [0 for i in range(mol.GetNumAtoms())]
        ring = mol.GetRingInfo()
        ring_atom_list = ring.AtomRings()
        for i in range(len(ring_atom_list)):
            for j in range(len(ring_atom_list)):
                if j > i:
                    overlap_atom =  list(set(ring_atom_list[i]).intersection(set(ring_atom_list[j])))
                    if len(overlap_atom) == 1:
                        SpiroList[overlap_atom[0]] = 1
                    if len(overlap_atom) > 1:
                        for atom in overlap_atom:
                            EndoList[atom] = 1
        return SpiroList,EndoList
    
    neighbor_num = get_BondNum(mol)
    RingNumList = get_inRingNum(mol)
    SpiroList,EndoList = get_SpiroEndo(mol)
    
    node_rdkit_merge = np.vstack((neighbor_num, RingNumList, SpiroList, EndoList))
    node_rdkit = node_rdkit_merge.T
    
    return node_rdkit


def atomic_para(atom_sym):
    if atom_sym not in ["H", "C", "N", "O", "F", "S", "Cl"]:
        if atom_sym == "Br":
            atom_sym = "Cl"
        else:
            atom_sym = "H"
    
    def atom_to_orbital(atom_sym):
        ## alpha: s p p p, beta: s p p p
        #diction = {'H': [-0.32169,0.0,0.0,0.0,0.02963,0.0,0.0,0.0],
        #           'C': [-0.58913,-0.26854,-0.26854,-0.15619,-0.47974,-0.12746,-0.08151,-0.08151],
        #           'O': [-1.02034,-0.46231,-0.46231,-0.38721,-0.86187,-0.33679,-0.18402,-0.18402],
        #           'N': [-0.81174,-0.35765,-0.35765,-0.35765,-0.61619,-0.11143,-0.11143,-0.11143],
        #           'F': [-1.23419,-0.5666,-0.47826,-0.47826,-1.14281,-0.44722,-0.44722,-0.27284],
        #           'S': [-0.72489,-0.33448,-0.33448,-0.28712,-0.63207,-0.26251,-0.17702,-0.17702],
        #           'Cl':[-0.84754,-0.4102,-0.35539,-0.35539,-0.79414,-0.34053,-0.34053,-0.24575]}
        # homo lumo
        diction = {'H': [-0.32169,0.02963],
                   'C': [-0.26854,-0.15619],
                   'O': [-0.33679,-0.18402],
                   'N': [-0.35765,-0.11143],
                   'F': [-0.44722,-0.27284],
                   'S': [-0.26251,-0.17702],
                   'Cl':[-0.34053,-0.24575]}
        return diction[atom_sym]
    
    def Element2Radius(atom_sym):
        #出于方便考虑C只用sp3杂化的，因为只会影响到格点数，应该问题不大
        Radius_Table = {"H":   0.31, "C":   0.76, "N":   0.70, "O":   0.66, "F":   0.57, "S":  1.05, "Cl":  1.02}
        return Radius_Table[atom_sym]
    
    def Electronegativity(atom_sym):
        Electronegativity_Table = {"H": 2.20, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98, "S": 2.58, "Cl": 3.16}
        return Electronegativity_Table[atom_sym]
    
    def Period(atom_sym):
        Period_dic = {'H': [1,0,0],
                      'C': [0,1,0],
                      'N': [0,1,0],
                      'O': [0,1,0],
                      'F': [0,1,0],
                      'S': [0,0,1],
                      'Cl':[0,0,1]
                            }
        return Period_dic[atom_sym]
    
    def Group(atom_sym):
        Group_dic = {'H':[1,0,0,0,0,0,0],
                     'C':[0,0,0,1,0,0,0],
                     'N':[0,0,0,0,1,0,0],
                     'O':[0,0,0,0,0,1,0],
                     'F':[0,0,0,0,0,0,1],
                     'S':[0,0,0,0,0,1,0],
                     'Cl':[0,0,0,0,0,0,1]
                    }
        return Group_dic[atom_sym]

    def Mass(atom_sym):
        Mass_dic = {'H': 1,
                    'C': 12,
                    'N': 14,
                    'O': 16,
                    'F': 19,
                    'S': 32,
                    'Cl':35.5
                        }
        return Mass_dic[atom_sym]
    
    def FirstIonEne(atom_sym):
        ##单位eV,bing搜索查得
        FirstIonEne_dic = {'H': 13.5984,
                           'C': 11.2603,
                           'N': 14.534,
                           'O': 13.6181,
                           'F': 17.4228,
                           'S': 10.36 ,
                           'Cl':12.9676
                               }
        return FirstIonEne_dic[atom_sym]
    
    def AtomEnergy(atom_sym):
        AtomHartree_dic = {'H': -0.495446,
                           'C': -37.752008,
                           'N': -54.451523,
                           'O': -74.925834,
                           'F': -99.680538,
                           'S': -397.978502,
                           'Cl':-460.067599
                              }
        return AtomHartree_dic[atom_sym]
    
    atom_orbitalE = atom_to_orbital(atom_sym)
    atom_radius = Element2Radius(atom_sym)
    atom_eg = Electronegativity(atom_sym)
    atom_per = Period(atom_sym)
    atom_gro = Group(atom_sym)
    
    atom_mass = Mass(atom_sym)
    atom_ion = FirstIonEne(atom_sym)
    atom_energy = AtomEnergy(atom_sym)
    atom_chempot=(atom_to_orbital(atom_sym)[1]+atom_to_orbital(atom_sym)[0])/2
    atom_hardness=(atom_to_orbital(atom_sym)[1]-atom_to_orbital(atom_sym)[0])/2

    return atom_orbitalE + [atom_radius, atom_eg] + atom_per + atom_gro + \
            [atom_mass,atom_ion,atom_energy,atom_chempot,atom_hardness]

