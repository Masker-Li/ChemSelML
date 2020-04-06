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
from ..bin.Label2Idx import Ar_dict_generator, R_dict_generator


def label2num(label, mode='Ar', src_dir=None):
    assert mode in ['Ar', 'R'], "mode should be Ar/R"
    
    if mode=='Ar':
        Ar_dict = Ar_dict_generator(src_dir=src_dir)
        return Ar_dict[label]
    else:
        R_dict = R_dict_generator(src_dir=src_dir)
        return R_dict[label]

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
        if self.suffix:
            return '%s/MolGraph_%s%s.pt' % (self.suffix, self.mode, suffix)
        else:
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
            mol_num = label2num(mol_name,self.mode,self.raw_paths[0])
        except KeyError:
            mol_name = mol_fn_ls[0]
            mol_num = label2num(mol_name,self.mode,self.raw_paths[0])
            
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
        processed_dir = os.path.dirname(self.processed_paths[0])
        if not os.path.isdir(processed_dir):
            os.makedirs(processed_dir)
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
