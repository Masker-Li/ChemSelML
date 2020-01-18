import os.path as osp
import os
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import remove_self_loops
import torch_geometric.transforms as T
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import networkx as nx
import pathlib
import ase
from ase.io import read
from ase import Atoms
import numpy as np
import pandas as pd
import copy
from molml.features import BagOfBonds

from ..bin.GaussianOutputFileReader import Gaussian_Output
from ..bin.featurization import get_acsf_features, get_CoulombMatrix, get_SOAP, PhyChem
from ..bin.featurization import get_RDKit, atomic_para


def label2num(label,mode='Ar'):
    assert mode in ['Ar', 'R'], "mode should be Ar/R"
    R_dict = {
        "CCN":  61 , "CF2":  62 , "CF3":  63 , "CFM":  64 , "CH3":  65 , 
        "CHF":  66 , "Et0":  67 , "ipr":  68 , "MF3":  69 ,  "Ph":  70 ,
        "Py0":  71 , "tBu":  72 , "TF3":  73 ,
        }
    
    Ar_dict = {
          "A0h" : 101 ,   "A1c" : 102 ,   "A1d" : 103 ,   "B0h" : 111 ,   "B1c" : 112 , 
          "B1d" : 113 ,   "B2c" : 114 ,   "B2d" : 115 ,   "C0h" : 121 ,   "C1c" : 122 , 
          "C1d" : 123 ,   "C2c" : 124 ,   "C2d" : 125 ,   "D0h" : 131 ,   "D1c" : 132 , 
          "D1d" : 133 ,   "D2c" : 134 ,   "D2d" : 135 ,   "E0h" : 141 ,   "E1c" : 142 , 
          "E1d" : 143 ,   "E2c" : 144 ,   "E2d" : 145 ,   "F0h" : 151 ,   "F1c" : 152 , 
          "F1d" : 153 ,   "F2c" : 154 ,   "F2d" : 155 ,   "F3c" : 156 ,   "F3d" : 157 , 
          "Fb0" : 161 ,  "Fb3c" : 162 ,  "Fb3d" : 163 ,  "Fb4c" : 164 ,  "Fb4d" : 165 , 
         "Fb5c" : 166 ,  "Fb5d" : 167 ,   "Fc0" : 171 ,  "Fc2c" : 172 ,  "Fc2d" : 173 , 
         "Fc4c" : 174 ,  "Fc4d" : 175 ,  "Fc5c" : 176 ,  "Fc5d" : 177 ,   "Fd0" : 181 , 
         "Fd3c" : 182 ,  "Fd3d" : 183 ,  "Fd4c" : 184 ,  "Fd4d" : 185 ,  "Fd5c" : 186 , 
         "Fd5d" : 187 ,   "Fe0" : 191 ,  "Fe2c" : 192 ,  "Fe2d" : 193 ,  "Fe4c" : 194 , 
         "Fe4d" : 195 ,  "Fe5c" : 196 ,  "Fe5d" : 197 ,   "Ff0" : 201 ,  "Ff3c" : 202 , 
         "Ff3d" : 203 ,  "Ff4c" : 204 ,  "Ff4d" : 205 ,  "Ff5c" : 206 ,  "Ff5d" : 207 , 
          "G0h" : 211 ,   "G1c" : 212 ,   "G1d" : 213 ,   "G2c" : 214 ,   "G2d" : 215 , 
          "G3c" : 216 ,   "G3d" : 217 ,  "G0hH" : 221 ,  "G1cH" : 222 ,  "G1dH" : 223 , 
         "G2cH" : 224 ,  "G2dH" : 225 ,  "G3cH" : 226 ,  "G3dH" : 227 ,   "H0h" : 231 , 
         "H01c" : 232 ,  "H01d" : 233 ,  "H0hH" : 241 , "H01cH" : 242 , "H01dH" : 243 , 
        "H0hH-c2" : 251 , "H01cH-c2" : 252 , "H01dH-c2" : 253 , 
         "HFa0" : 261 , "HFa2c" : 262 , "HFa2d" : 263 , "HFa4c" : 264 , "HFa4d" : 265 , 
        "HFa5c" : 266 , "HFa5d" : 267 ,  "HFb0" : 271 , "HFb3c" : 272 , "HFb3d" : 273 , 
        "HFb4c" : 274 , "HFb4d" : 275 , "HFb5c" : 276 , "HFb5d" : 277 ,  "HFc0" : 281 , 
        "HFc2c" : 282 , "HFc2d" : 283 , "HFc4c" : 284 , "HFc4d" : 285 , "HFc5c" : 286 , 
        "HFc5d" : 287 ,  "HFd0" : 291 , "HFd3c" : 292 , "HFd3d" : 293 , "HFd4c" : 294 , 
        "HFd4d" : 295 , "HFd5c" : 296 , "HFd5d" : 297 ,  "HFe0" : 301 , "HFe2c" : 302 ,
        "HFe2d" : 303 , "HFe4c" : 304 , "HFe4d" : 305 , "HFe5c" : 306 , "HFe5d" : 307 ,
         "HFf0" : 311 , "HFf3c" : 312 , "HFf3d" : 313 , "HFf4c" : 314 , "HFf4d" : 315 ,
        "HFf5c" : 316 , "HFf5d" : 317 ,   "I0h" : 321 ,   "I1c" : 322 ,   "I1d" : 323 ,
         "I0hH" : 331 ,  "I1cH" : 332 ,  "I1dH" : 333 ,
        "I0hH-c2" : 341 , "I1cH-c2" : 342 , "I1dH-c2" : 343 ,
            "J0h" : 361,  "I1cH-c3" : 352 , "I1dH-c3" : 353,
        
        "O2a": 401, "O2aH": 402,  "P4b": 411,  "P4bH": 412,  "P4c": 413, 
        "P4cH": 414,  "P4d": 415,  "P4dH": 416,  "P4e": 417,  "P4eH": 418, 
        "P4f": 419,  "P4fH": 420,  "Q1a": 421,  "R4g": 431,  "R4gH": 432, 
        "R4gH-c2": 433,  "R4h": 434, "R4hH": 435, "R4hH-c2": 436,
        "R4i": 437,  "R4iH": 438,  "R4iH-c2": 439,  "S1b": 441,  "S1h": 442, 
        "T2j": 451,  "T2jH": 452,  "T2jH-c2": 453,  "U2k": 461,  "U2kH": 462, 
        
        "D1f": 501, "D2f": 502, "G1n": 511, "G1nH": 512, "G2g": 513,
        "G2gH": 514, "G2m": 515, "G2mH": 516, "G2n": 517, "G2nH": 518,
        "G3c": 519, "G3cH": 520, "G3e": 521, "G3eH": 522, "G3f": 523,
        "G3fH": 524, "K1e": 531, "K1eH": 532, "L1d": 541, "O4n": 551,
        "O4nH": 552, "O4t": 553, "O4tH": 554, "O5n": 555, "O5nH": 556,
        "P7s": 561, "P7sH": 562,
        }
    return R_dict[label] if mode == 'R' else Ar_dict[label]

class BaseDataset(InMemoryDataset):
    fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

    def __init__(self, root, mode='Ar', suffix=None, transform=None, pre_transform=None, pre_filter=None):
        self.mode = mode
        self.root = root
        self.suffix = suffix 
        assert mode in ['Ar', 'R'], "mode should be Ar/R"
        super(BaseDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['%s/%s'%(self.suffix, self.mode), ] if self.suffix else [self.mode, ]

    @property
    def processed_file_names(self):
        suffix = '_%s'%self.suffix if self.suffix else None
        return 'MolGraph_%s%s.pt' % (self.mode, suffix)

    def download(self):
        return 0
        raise NotImplementedError('please download and unzip dataset from %s, and put it at %s' % (_urls[self.mode], self.raw_dir))
    
    def mol_nodes(self, g):
        feat, feat_SF = [], []
        for n, d in g.nodes(data=True):
            h_t, SF_t = [], []
            # Atom type (One-hot H, C, N, O F)
            h_t += [int(d['a_type'] == x) for x in ['H', 'C', 'N', 'O', 'F', 'S']]
            # Atomic number
            h_t.append(d['a_num'])
            SF_t = copy.copy(h_t)
            # Acceptor
            h_t.append(d['acceptor'])
            # Donor
            h_t.append(d['donor'])
            # Aromatic
            h_t.append(int(d['aromatic']))
            # Hybradization
            h_t += [int(d['hybridization'] == x) \
                    for x in (Chem.rdchem.HybridizationType.SP, \
                        Chem.rdchem.HybridizationType.SP2,
                        Chem.rdchem.HybridizationType.SP3)]
            h_t.append(d['num_h'])
            ##新增2行
            h_t.append(d['NPA_chg'])
            h_t += list(d['rdkit_merge'])
            # add other atom parameter
            h_t += atomic_para(d['a_type'])
            # add SymmetryFunction
            SF_t += d['SymmetryFunction']
            feat.append((n, h_t))
            feat_SF.append((n,SF_t))
            
        feat.sort(key=lambda item: item[0])
        node_attr = torch.FloatTensor([item[1] for item in feat])
        feat_SF.sort(key=lambda item: item[0])
        node_SF = torch.FloatTensor([item[1] for item in feat_SF])
        return node_attr, node_SF

    def mol_edges(self, g):
        e={}
        h=g.to_undirected()
        flag=1
        try:
            p=list(nx.simple_cycles(h))
        except :
            flag=0
        #for n1, n2, d in g.edges(data=True):
        for N1 in g.nodes(data=True):
            n1=N1[0]
            ch1=N1[1]['NPA_chg']
            for N2 in g.nodes(data=True):
                n2=N2[0]
                ch2=N2[1]['NPA_chg']
                ch=ch1*ch2
                flag2=0
                try: 
                     y=g.edges[n1,n2]['b_type']
                except:
                    e_t =[0,0,0,0]
                else:
                    e_t = [int(y == x)
                            for x in (Chem.rdchem.BondType.SINGLE, \
                            Chem.rdchem.BondType.DOUBLE, \
                            Chem.rdchem.BondType.TRIPLE, \
                            Chem.rdchem.BondType.AROMATIC) ]
        

                if flag==1:
                    for i in range (len(p)):
                        if n1 in p[i] and n2 in p[i]:
                            flag2=1
                            break
                e_t.append(flag2)
                try: 
                    di=int(nx.algorithms.shortest_paths.generic.shortest_path_length(h,n1,n2))
                except:
                    di=0
                e_t.append(di)
                e_t.append(ch)
                e[(n1, n2)] = e_t
        edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
        edge_attr = torch.FloatTensor(list(e.values()))
        return edge_index, edge_attr

    # gaussian file reader for Mol dataset                 
    def gaussian_graph_reader(self, mol_file):
        GO = Gaussian_Output(self.root, mol_file, 'PRE')
        mol = GO.rdkitmol
        keyAtoms = np.array(GO.Atom_labels_list) - 1
        
        tmp_atoms = read(mol_file,format='gaussian-out')
        ase_atoms = Atoms(tmp_atoms.symbols, GO.AtomsCoordinates)
        ase_atoms.set_initial_charges(GO.NPACharge)
        ase_atoms.set_atomic_numbers(GO.AtomsNum)
        tmp_mol = ([ase_atoms.numbers, ase_atoms.positions])
        if 'NPACharges' not in ase_atoms.arrays.keys():
            ase_atoms.new_array('NPACharges',GO.AtomsNum + GO.NPACharge)
        ascf_features = get_acsf_features(ase_atoms)
        ascf_features_local = get_acsf_features(ase_atoms,list(keyAtoms))
        CM_features = get_CoulombMatrix(ase_atoms)
        SOAP_features = get_SOAP(ase_atoms,list(keyAtoms))

        if mol is None:
            print("rdkit can not parsing", mol_file)
            return None
        feats = self.chem_feature_factory.GetFeaturesForMol(mol)
        MACCSfp = AllChem.GetMACCSKeysFingerprint(mol)
        MACCSfp = torch.ByteTensor([int(x) for x in MACCSfp.ToBitString()])
        Morganfp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=512)
        Morganfp = torch.ByteTensor([int(x) for x in Morganfp.ToBitString()])
        
        g = nx.DiGraph()
        bond_ring_info = get_RDKit(mol)        
        
        # for training set, we store its target
        # otherwise, we store its molecule id
        mol_fn_ls = mol_file.stem.split('-')
        if self.mode == "Ar":
            mol_name = mol_fn_ls[0] if mol_fn_ls[2] == 'c1' else mol_fn_ls[0] + "-" + mol_fn_ls[2]
        elif self.mode == "R":
            mol_name = mol_fn_ls[0]
        
        try:
            mol_num = label2num(mol_name,mode=self.mode)
        except KeyError:
            mol_name = mol_fn_ls[0]
            mol_num = label2num(mol_name,mode=self.mode)
            
        l = torch.FloatTensor(self.target.loc[int(mol_num)].tolist()).unsqueeze(0) \
                if self.mode == 'dev' else torch.LongTensor([int(mol_num)])
        alias = torch.LongTensor([int(mol_num)])
        
        # add PhyChem descriptors
        _idx = int(mol_num)*10 + keyAtoms if self.mode == 'Ar' else np.array([int(mol_num)])
        phy_chem_total = self.PhyChem.get_total_PhyChem(_idx[0])
        phy_chem_local = self.PhyChem.get_local_PhyChem(_idx)
        
        charge_list = GO.AtomsNum + GO.NPACharge

        # Create nodes
        assert len(mol.GetConformers()) == 1
        geom = mol.GetConformers()[0].GetPositions()

        for i in range(mol.GetNumAtoms()):
            atom_i = mol.GetAtomWithIdx(i)
            g.add_node(i, a_type=atom_i.GetSymbol(), a_num=atom_i.GetAtomicNum(), acceptor=0, donor=0,
                aromatic=atom_i.GetIsAromatic(), hybridization=atom_i.GetHybridization(),
                   num_h=atom_i.GetTotalNumHs(),NPA_chg=charge_list[i],rdkit_merge=bond_ring_info[i],
                      SymmetryFunction=list(ascf_features[i]))

        for i in range(len(feats)):
            if feats[i].GetFamily() == 'Donor':
                node_list = feats[i].GetAtomIds()
                for i in node_list:
                    g.node[i]['donor'] = 1
            elif feats[i].GetFamily() == 'Acceptor':
                node_list = feats[i].GetAtomIds()
                for i in node_list:
                    g.node[i]['acceptor'] = 1
        
        # Read Edges
        for i in range(mol.GetNumAtoms()):
            for j in range(mol.GetNumAtoms()):
                e_ij = mol.GetBondBetweenAtoms(i, j)
                if e_ij is not None:
                    g.add_edge(i, j, b_type=e_ij.GetBondType())

        
        node_attr, node_SF = self.mol_nodes(g)
        edge_index, edge_attr = self.mol_edges(g)
        data = Data(
                x=node_attr,
                pos=torch.FloatTensor(geom),
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=l
                )
        data.ACSF = node_SF
        data.ACSF_local = torch.FloatTensor(ascf_features_local)
        data.alias = alias
        data.PhyChem_total = torch.FloatTensor(phy_chem_total.values).unsqueeze(0)
        data.PhyChem_local = torch.FloatTensor(phy_chem_local.values)
        data.SOAP = torch.FloatTensor(SOAP_features)
        data.CM = torch.FloatTensor(CM_features)
        data.MACCSfp = MACCSfp.unsqueeze(0)
        data.Morganfp = Morganfp.unsqueeze(0)
        data.mergefp = torch.cat((MACCSfp,Morganfp)).unsqueeze(0)
        data.keyAtom_list = torch.LongTensor(list([([int(mol_num)]*len(keyAtoms)),list(keyAtoms)])).transpose(1,0)
        # data.BoB in self.process
        return data, tmp_mol

    def process(self):
        '''
        if self.mode == 'dev':
            self.target = pd.read_csv(self.raw_paths[1], index_col=0,
                    usecols=['gdb_idx',] + ['property_%d' % x for x in range(12)])
            self.target = self.target[['property_%d' % x for x in range(12)]]
        '''
        gaussian_dir = pathlib.Path(self.raw_paths[0])
        self.PhyChem = PhyChem(mode=self.mode, suffix=self.suffix)
        self.PhyChem_labels_total = self.PhyChem.total_PhyChem_labels
        self.PhyChem_labels_local = self.PhyChem.local_PhyChem_labels
        data_list, mol_list = [], []
        for mol_file in gaussian_dir.glob("**/*sp.log"):
            mol_data, tmp_mol = self.gaussian_graph_reader(mol_file)
            if mol_data is not None:
                data_list.append(mol_data)
                mol_list.append(tmp_mol)
        
        # add BoB descriptor
        BoB = BagOfBonds()
        BoB.fit(mol_list)
        self.BOB_labels = BoB.get_bob_labels(BoB._bag_sizes)
        for i in range(len(mol_list)):
            tmp_BoB = BoB.transform([mol_list[i]])
            data_list[i].BoB = torch.LongTensor(tmp_BoB)
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        
class ElectroNegativityDiff(object):

    def __init__(self, norm=True, max_value=None, cat=True):
        self.norm = norm
        self.max = max_value
        self.cat = cat
    def elec_nega(self,atom_pair):
        atom_1,atom_2 = atom_pair
        elec_nega_table = { 0: 2.20,      #H
                            1: 2.55,      #C
                            2: 3.04,      #N
                            3: 3.44,      #O
                            4: 3.98,      #F
                            5: 2.58,      #S
                            6: 3.16,      #Cl
                            }
        elec_nega_1,elec_nega_2 = elec_nega_table[atom_1],elec_nega_table[atom_2]
        Elec_diff = elec_nega_1 - elec_nega_2         #电负性差值
        return Elec_diff    
    def elec_dire(self,elec_diff):
        if elec_diff < 0:
            return np.array([0,1])
        elif elec_diff > 0:
            return np.array([1,0])
        else:
            return np.array([0,0])
    def __call__(self, data):
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr
        atom_type = data.x[:,:7]
        atom_type_x,atom_type_y = atom_type[row],atom_type[col]
        
        dist = torch.norm(pos[col] - pos[row], p=2, dim=-1).view(-1, 1)
        
        atom_pair_stack = np.stack((atom_type_x.argmax(dim=1),atom_type_y.argmax(dim=1)),axis=1)
        temp_elec_diff = np.array([self.elec_nega(atom_pair) for atom_pair in atom_pair_stack],dtype=np.float32)
        
        elec_dire_one_hot = torch.tensor([self.elec_dire(elec_diff) for elec_diff in temp_elec_diff],dtype=torch.float32).view(-1,2)
        elec_diff = torch.tensor(np.abs(temp_elec_diff),dtype=torch.float32).view(-1,1)
        if self.norm and dist.numel() > 0:
            dist = dist / dist.max() if self.max is None else self.max

        
        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, dist.type_as(pseudo),elec_diff,elec_dire_one_hot], dim=-1)
        else:
            data.edge_attr = torch.cat([dist,elec_diff,elec_dire_one_hot],dim=-1)

            
            
        return data

    def __repr__(self):
        return '{}(norm={}, max_value={},writer: Licheng Xu)'.format(self.__class__.__name__,
                                                  self.norm, self.max)

class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data
