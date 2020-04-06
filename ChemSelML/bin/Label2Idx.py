import os
import sys
import pandas as pd
import numpy as np
import json
from bidict import bidict


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

        
def Ar_dict_generator(src_dir, mode="test"):
    if os.path.isfile('%s/Ar_dict.json'%src_dir):
        with open('%s/Ar_dict.json'%src_dir, 'r') as f:
            Ar_dict = json.load(f)
    else:
        df = pd.read_csv(src_dir+'/Ar_phychem.csv',index_col=0)
        if set(['Ar_loc_idx', 'Ar_idx', 'loc_idx']).issubset(df.columns):
            ks, vs = df['Ar_n'], df['Ar_idx']
            Ar_dict = {ks[i]: vs[i] for i in df.index}
        else:
            assert mode in ['dev', 'valid', 'test'], "mode should be dev/valid/test"
            start_num = 101 if mode == "dev" else 401
            Ar_list = sorted(list(set(df['Ar_n'])))
            Ar_dict = {k: v+start_num for v, k in enumerate(Ar_list)}
            
            df.insert(3, 'Ar_idx', df['Ar_n'].apply(lambda x: Ar_dict[x]))
            df.insert(4, 'loc_idx', df['loc_n'].astype(float)-1)
            df.insert(3, 'Ar_loc_idx', df['Ar_idx']*10 + df['loc_idx'])
            df.to_csv(src_dir+'/Ar_phychem.csv')
        
        json_str = json.dumps(Ar_dict, indent=4, cls=NpEncoder)
        with open('%s/Ar_dict.json'%src_dir, 'w') as json_file:
            json_file.write(json_str)
    return Ar_dict


def R_dict_generator(src_dir):
    if os.path.isfile('%s/R_dict.json'%src_dir):
        with open('%s/R_dict.json'%src_dir, 'r') as f:
            R_dict = json.load(f)
        return R_dict
    elif not os.path.isfile('%s/R_phychem.csv'%src_dir):
        R_dict = {
        "CCN":  61 , "CF2":  62 , "CF3":  63 , "CFM":  64 , "CH3":  65 , 
        "CHF":  66 , "Et0":  67 , "ipr":  68 , "MF3":  69 ,  "Ph":  70 ,
        "Py0":  71 , "tBu":  72 , "TF3":  73 ,
        }
    
    else:
        df = pd.read_csv(src_dir+'/R_phychem.csv',index_col=0)
        if set(['R_idx']).issubset(df.columns) :
            ks, vs = df['R_n'], df['R_idx']
            R_dict = {ks[i]: vs[i] for i in df.index}
        else:
            R_list = list(set(df['R_n']))
            R_dict = {k: v+61 for v, k in enumerate(R_list)}
            
            df.insert(1, 'R_idx', df['R_n'].apply(lambda x: R_dict[x]))
            df.to_csv(src_dir+'/R_phychem.csv')
        
    json_str = json.dumps(R_dict, indent=4, cls=NpEncoder)
    with open('%s/R_dict.json'%src_dir, 'w') as json_file:
        json_file.write(json_str)
    return R_dict


def ArR_label_Idx_Insertion(src_f):
    dict_src_path = os.path.dirname(src_f)
    Ar_dict, R_dict = get_Ar_R_dict(dict_src_path)
    
    df = pd.read_csv(src_f,index_col=0)
    if set(['Reaction_idx', 'Radical_idx', 'Ar_idx', 'loc_idx']).issubset(df.columns) :
        print('ArR_label_Idx has already been ready!')
        return
    else:
        n_col = df.columns.shape[0]
        df.insert(n_col-1, 'Radical_idx', df['Radical_n'].apply(lambda x: R_dict[x]))
        df.insert(n_col, 'Ar_idx', df['Ar_n'].apply(lambda x: Ar_dict[x]))
        df.insert(n_col+1, 'loc_idx', df['loc_n'].astype(float)-1)
        df.insert(n_col-1, 'Reaction_idx', df['Radical_idx']*10000+df['Ar_idx']*10+df['loc_idx'])
        df.to_csv(src_f)


def get_Ar_R_dict(dict_src_path, reverse_dict=False):
    Ar_src_dir = "%s/%s"%(dict_src_path, 'Ar')
    R_src_dir = "%s/%s"%(dict_src_path, 'R')
    Ar_dict = Ar_dict_generator(src_dir=Ar_src_dir)
    R_dict = R_dict_generator(src_dir=R_src_dir)
    if not reverse_dict:
        return Ar_dict, R_dict
    else:
        bi_Ar_dict = bidict(Ar_dict)
        Ar_dict_inverse = bi_Ar_dict.inverse
        bi_R_dict = bidict(R_dict)
        R_dict_inverse = bi_R_dict.inverse
        return Ar_dict_inverse, R_dict_inverse
