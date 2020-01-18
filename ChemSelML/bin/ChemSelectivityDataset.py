import os.path as osp
import os
import torch
import numpy as np
import pandas as pd
import math
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import torch_geometric.transforms as T

from ..bin.BaseDataset import BaseDataset, ElectroNegativityDiff, Complete

transform = T.Compose([Complete(), ElectroNegativityDiff(norm=False)])

class ReactionDataset(InMemoryDataset):
    def __init__(self, root, mode='dev', transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.mode = mode
        assert mode in ['dev', 'valid',
                        'test'], "mode should be dev/valid/test"
        super(ReactionDataset, self).__init__(
            root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        if self.mode == 'dev':
            return [self.mode, '%s/TrainSet_Labels.csv'%self.mode]
        elif self.mode == 'valid':
            return [self.mode, '%s/ValidSet_Labels.csv'%self.mode]
        else:
            return [self.mode, '%s/TestSet_Labels.csv'%self.mode]

    @property
    def processed_file_names(self):
        return 'ReactionDataset_ArR_%s.pt' % self.mode

    def download(self):
        return 0
        #raise NotImplementedError('please download and unzip dataset from %s, and put it at %s' % (
        #    _urls[self.mode], self.raw_dir))

    def base_data_index_collector(self):
        R_dataset = BaseDataset(root=self.root, mode='R', suffix=self.mode, pre_transform=transform)
        Ar_dataset = BaseDataset(root=self.root, mode='Ar', suffix=self.mode, pre_transform=transform)
        self.R_dataset, self.Ar_dataset = R_dataset, Ar_dataset

        target = pd.read_csv(self.raw_paths[1], index_col=0,
                             usecols=['Reaction_idx', 'Radical_idx', 'Ar_idx', 'loc_idx', 'DG_TS'])
        origin_Ar_loc = Ar_dataset.data.keyAtom_list[:,
                             0]*10 + Ar_dataset.data.keyAtom_list[:, 1]
        tmp_Ar_loc = list(target['Ar_idx']*10 + target['loc_idx'])
        self.Ar_loc_index = (origin_Ar_loc == torch.LongTensor(
            tmp_Ar_loc).unsqueeze(0).transpose(1, 0)).nonzero()[:, 1]
        self.Ar_index = (Ar_dataset.data.alias == torch.LongTensor(
            list(target['Ar_idx'])).unsqueeze(0).transpose(1, 0)).nonzero()[:, 1]
        self.R_index = (R_dataset.data.alias == torch.LongTensor(
            list(target['Radical_idx'])).unsqueeze(0).transpose(1, 0)).nonzero()[:, 1]
        self.target = target[['DG_TS']]
        self.target_labels = ['DG_TS']

    def reaction_reader(self, i, ArR_idx):
        '''
        i : index of ArR_idx in ArR ReactionDataset
        ArR_idx: Reaction_idx in ArR ReactionDataset, 
                form: 6 digit integer，
                [R_idx:2][Ar_idx:3][loc_idx:1]
        y: DG_TS in ArR ReactionDataset
        '''
        local_f = ['ACSF_local', 'SOAP', 'PhyChem_local']
        total_f = ['MACCSfp', 'Morganfp', 'mergefp', 'PhyChem_total',
                   'ACSF', 'pos', 'x', 'edge_attr', 'CM', 'BoB']
        other_f = ['edge_index']

        DG_TS = torch.FloatTensor(self.target.iloc[i].tolist())

        data = Data(y=DG_TS)
        data.ArR_idx = ArR_idx

        for f in local_f:
            Arl_i = self.Ar_loc_index[i]
            Arl_j = self.Ar_loc_index[i]+1
            data['Ar_'+f] = self.Ar_dataset.data[f][Arl_i:Arl_j, :]
            R_i = self.R_dataset.slices[f][self.R_index[i]]
            R_j = self.R_dataset.slices[f][self.R_index[i]+1]
            data['R_'+f] = self.R_dataset.data[f][R_i:R_j, :]
        
        for f in total_f:
            Ar_i = self.Ar_dataset.slices[f][self.Ar_index[i]]
            Ar_j = self.Ar_dataset.slices[f][self.Ar_index[i]+1]
            data['Ar_'+f] = self.Ar_dataset.data[f][Ar_i:Ar_j, :]
            R_i = self.R_dataset.slices[f][self.R_index[i]]
            R_j = self.R_dataset.slices[f][self.R_index[i]+1]
            data['R_'+f] = self.R_dataset.data[f][R_i:R_j, :]

        for f in other_f:
            Ar_i = self.Ar_dataset.slices[f][self.Ar_index[i]]
            Ar_j = self.Ar_dataset.slices[f][self.Ar_index[i]+1]
            data['Ar_'+f] = self.Ar_dataset.data[f][:, Ar_i:Ar_j]
            R_i = self.R_dataset.slices[f][self.R_index[i]]
            R_j = self.R_dataset.slices[f][self.R_index[i]+1]
            data['R_'+f] = self.R_dataset.data[f][:, R_i:R_j]

        return data

    def process(self):
        self.base_data_index_collector()
        data_list = []
        for i, ArR_idx in enumerate(self.target.index):
            ArR_data = self.reaction_reader(i, ArR_idx)
            if ArR_data is not None:
                data_list.append(ArR_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        
def Transform_DG_to_DDG(TrainSet, scale_ref=5, verbose=True):
    def Part_select(a, b, which_part):
        assert which_part in ['up', 'down'], "which_part should be up/down"
        if which_part == 'up':
            return a > b
        elif which_part == 'down':
            return a < b

    def get_Scale(DDG):
        # scale: @A vs @B
        scale = np.piecewise(DDG, [DDG >= 0, DDG < 0], [lambda x: math.exp(-x*4.184*1000/(
            298.15*8.31451)), lambda x: -math.exp(x*4.184*1000/(298.15*8.31451))])
        return scale

    def get_Type(scale, scale_ref=5):
        Type = np.piecewise(scale, [scale < -1/scale_ref, -1/scale_ref <= scale < -(1/scale_ref)**2,
                                    -(1/scale_ref)**2 <= scale < 0, 0 < scale <= (1/scale_ref)**2,
                                    (1/scale_ref)**2 < scale <= 1/scale_ref, 1/scale_ref < scale],
                            [0, -1, -2, 2, 1, 0])
        return Type

    Group_data = TrainSet.groupby(['Ar_idx', 'loc_idx', 'Radical_idx'])[
        'DG_TS'].mean().unstack()
    Ars = Group_data.index.levels[0]
    Radicals = Group_data.columns

    def get_DDG(which_part='up'):
        DDG_TS_ls = []
        for Ar in Ars:
            for R in Radicals:
                loc = Group_data.loc[Ar, R].index
                for loc_A in loc:
                    for loc_B in loc:
                        if Part_select(loc_B, loc_A, which_part):
                            DG_TS_A = Group_data.loc[Ar, loc_A][R]
                            DG_TS_B = Group_data.loc[Ar, loc_B][R]
                            if pd.notna(DG_TS_A) == True and pd.notna(DG_TS_B) == True:
                                DDG = DG_TS_B - DG_TS_A
                                DDG_TS_ls = DDG_TS_ls + \
                                    [[Ar, R, loc_A, loc_B, np.round(
                                        DDG, 2), DG_TS_A, DG_TS_B]]

        DDG_TS_columns = ['Ar', 'Radical', 'locA',
                          'locB', 'DDG_TS', 'DG_TS_@A', 'DG_TS_@B']
        DDG_TS_pd = pd.DataFrame(DDG_TS_ls, columns=DDG_TS_columns)
        DDG_TS_pd['ArR_sel_idx'] = DDG_TS_pd['Radical']*100000 + \
            DDG_TS_pd['Ar']*100 + DDG_TS_pd['locA']*10 + DDG_TS_pd['locB']*1
        if verbose:
            print('DDG_TS_pd:', DDG_TS_pd.shape)

        DDG_TS_pd['scale_AvsB'] = DDG_TS_pd['DDG_TS'].apply(
            get_Scale)
        DDG_TS_pd['Type'] = DDG_TS_pd['scale_AvsB'].apply(
            get_Type)
        #Old_Columns = list(DDG_TS_pd.columns)
        #New_Columns = Old_Columns[:5] + Old_Columns[-2:] + Old_Columns[5:-2]
        #DDG_TS_pd = DDG_TS_pd[New_Columns]

        return DDG_TS_pd

    DDG_up = get_DDG(which_part='up')
    DDG_up['Descending'] = 0
    DDG_up.set_index('ArR_sel_idx', inplace=True)
    
    DDG_down = get_DDG(which_part='down')
    DDG_down['Descending'] = 1
    DDG_down['ArR_sel_idx'] = DDG_down['ArR_sel_idx'] + 10000000
    DDG_down.set_index('ArR_sel_idx', inplace=True)
    
    DDG_TS = pd.concat([DDG_up, DDG_down],ignore_index=False)
    return DDG_TS


class SelectivityDataset(InMemoryDataset):
    def __init__(self, root, mode='dev', transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.mode = mode
        assert mode in ['dev', 'valid',
                        'test'], "mode should be dev/valid/test"
        super(SelectivityDataset, self).__init__(
            root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        if self.mode == 'dev':
            return [self.mode, '%s/TrainSet_Labels.csv'%self.mode]
        elif self.mode == 'valid':
            return [self.mode, '%s/ValidSet_Labels.csv'%self.mode]
        else:
            return [self.mode, '%s/TestSet_Labels.csv'%self.mode]

    @property
    def processed_file_names(self):
        return 'SelectivityDataset_ArR_%s.pt' % self.mode

    def download(self):
        return 0
        # raise NotImplementedError('please download and unzip dataset from %s, and put it at %s' % (
        #    _urls[self.mode], self.raw_dir))

    def base_data_index_collector(self, part_select='up'):
        R_dataset = BaseDataset(root=self.root, mode='R', suffix=self.mode, pre_transform=transform)
        Ar_dataset = BaseDataset(root=self.root, mode='Ar', suffix=self.mode, pre_transform=transform)
        self.R_dataset, self.Ar_dataset = R_dataset, Ar_dataset

        target = pd.read_csv(self.raw_paths[1], index_col=0,
                             usecols=['Reaction_idx', 'Radical_idx', 'Ar_idx', 'loc_idx', 'DG_TS'])
        DDG_TS = Transform_DG_to_DDG(target, scale_ref=5)
        DDG_TS.to_csv(r'%s/TrainSet_DDG_Lables.csv'%self.processed_dir)
        
        origin_Ar_loc = Ar_dataset.data.keyAtom_list[:,
                             0]*10 + Ar_dataset.data.keyAtom_list[:, 1]
        tmp_Ar_locA = list(DDG_TS['Ar']*10 + DDG_TS['locA'])
        self.Ar_locA_index = (origin_Ar_loc == torch.LongTensor(
            tmp_Ar_locA).unsqueeze(0).transpose(1, 0)).nonzero()[:, 1]
        tmp_Ar_locB = list(DDG_TS['Ar']*10 + DDG_TS['locB'])
        self.Ar_locB_index = (origin_Ar_loc == torch.LongTensor(
            tmp_Ar_locB).unsqueeze(0).transpose(1, 0)).nonzero()[:, 1]
        self.Ar_index = (Ar_dataset.data.alias == torch.LongTensor(
            list(DDG_TS['Ar'])).unsqueeze(0).transpose(1, 0)).nonzero()[:, 1]
        self.R_index = (R_dataset.data.alias == torch.LongTensor(
            list(DDG_TS['Radical'])).unsqueeze(0).transpose(1, 0)).nonzero()[:, 1]
        self.target = DDG_TS[['DDG_TS', 'Type', 'Descending']]
        self.target_labels = ['DDG_TS', 'Type', 'Descending']
        

    def reaction_reader(self, i, ArR_sel_idx):
        '''
        i : index of ArR_sel_idx in ArR ReactionDataset
        ArR_sel_idx: Reaction_idx in ArR ReactionDataset, 
                form: 6 digit integer，
                [R_idx:2][Ar_idx:3][loc_idx:1]
        y: DDG_TS, Type and Descending in ArR ReactionDataset
        '''
        local_f = ['ACSF_local', 'SOAP', 'PhyChem_local']
        total_f = ['MACCSfp', 'Morganfp', 'mergefp', 'PhyChem_total',
                   'ACSF', 'pos', 'x', 'edge_attr', 'CM', 'BoB']
        other_f = ['edge_index']
        self.local_f = local_f
        self.total_f = total_f

        DDG_TS = torch.FloatTensor(self.target.iloc[i].tolist()).unsqueeze(0)

        data = Data(y=DDG_TS)
        data.ArR_sel_idx = ArR_sel_idx

        for f in local_f:
            Arl_i = self.Ar_locA_index[i]
            Arl_j = self.Ar_locA_index[i]+1
            data['Ar_'+f+'@A'] = self.Ar_dataset.data[f][Arl_i:Arl_j, :]
            Arl_i = self.Ar_locB_index[i]
            Arl_j = self.Ar_locB_index[i]+1
            data['Ar_'+f+'@B'] = self.Ar_dataset.data[f][Arl_i:Arl_j, :]
            R_i = self.R_dataset.slices[f][self.R_index[i]]
            R_j = self.R_dataset.slices[f][self.R_index[i]+1]
            data['R_'+f] = self.R_dataset.data[f][R_i:R_j, :]

        for f in total_f:
            Ar_i = self.Ar_dataset.slices[f][self.Ar_index[i]]
            Ar_j = self.Ar_dataset.slices[f][self.Ar_index[i]+1]
            data['Ar_'+f] = self.Ar_dataset.data[f][Ar_i:Ar_j, :]
            R_i = self.R_dataset.slices[f][self.R_index[i]]
            R_j = self.R_dataset.slices[f][self.R_index[i]+1]
            data['R_'+f] = self.R_dataset.data[f][R_i:R_j, :]

        for f in other_f:
            Ar_i = self.Ar_dataset.slices[f][self.Ar_index[i]]
            Ar_j = self.Ar_dataset.slices[f][self.Ar_index[i]+1]
            data['Ar_'+f] = self.Ar_dataset.data[f][:, Ar_i:Ar_j]
            R_i = self.R_dataset.slices[f][self.R_index[i]]
            R_j = self.R_dataset.slices[f][self.R_index[i]+1]
            data['R_'+f] = self.R_dataset.data[f][:, R_i:R_j]

        return data

    def process(self):
        self.base_data_index_collector()
        data_list = []
        for i, ArR_sel_idx in enumerate(self.target.index):
            ArR_data = self.reaction_reader(i, ArR_sel_idx)
            if ArR_data is not None:
                data_list.append(ArR_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
