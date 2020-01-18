import math
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd

from ChemSelML.bin.ChemSelectivityDataset import ReactionDataset, SelectivityDataset
from ChemSelML.bin.ChemSelectivityDataset import Transform_DG_to_DDG

ArR_dataset = ReactionDataset(root='/PyScripts/PyTorch.dir/Radical/DataSet', mode='dev')
ArR_DDG_dataset = SelectivityDataset(root='/PyScripts/PyTorch.dir/Radical/DataSet', mode='dev')

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

def reg2clf(y_reg):
    y_reg = y_reg.reshape(1,-1)
    y_scale = np.apply_along_axis(get_Scale, 0, y_reg)
    y_clf = np.apply_along_axis(get_Type, 0, y_scale)
    y_clf = y_clf[0].astype(dtype=np.int32)
    return y_clf

def class_5to3(y_5c):
    # copy y_5c
    y_3c = y_5c.copy()
    # change y_5c: 5-class to 3-class [-2,-1:0:1,2]
    y_3c[y_3c < 0] = -1
    y_3c[y_3c == 0] = 0
    y_3c[y_3c > 0] = 1
    return y_3c

def DG_to_DDG(idx, DG_TS):
    pred_dict = {'idx': idx, 'DG_TS': DG_TS}
    df = pd.DataFrame.from_dict(pred_dict)
    columns = list(df.columns)
    # S R Ar loc1 loc2 for X XX XXX X X
    df['Radical_idx'] = df['idx']//10000
    # S R Ar loc1 loc2 for X XX XXX X X
    df['Ar_idx'] = df['idx'] % 10000//10
    # S R Ar loc1 loc2 for X XX XXX X X
    df['loc_idx'] = df['idx'] % 10
    columns = columns[0:1] + ['Ar_idx', 'Radical_idx', 'loc_idx'] + columns[1:]
    df = df[columns]
    df.set_index('idx', inplace=True)
    DDG_df = Transform_DG_to_DDG(df, verbose=False)
    DDG_TS = DDG_df['DDG_TS'].to_numpy()
    Type = DDG_df['Type'].to_numpy()
    return DDG_TS, Type

def reg2clf_Evaluation(y_true, y_pred, idx, mode='DDG_R'):
    assert mode in ['DG_R', 'DDG_R',
                    'DDG_C'], 'mode should in DG_R/DDG_R/DDG_C'
    if mode == 'DDG_R':
        # 5 class
        y_true_reg, y_pred_reg = y_true, y_pred
        y_true_clf_5c = reg2clf(y_true_reg)
        y_pred_clf_5c = reg2clf(y_pred_reg)
        F1_micro_5c = f1_score(y_true_clf_5c, y_pred_clf_5c, average='micro')
        F1_micro_5c = np.round(F1_micro_5c, decimals=4)
        # 3 class
        y_true_clf_3c = class_5to3(y_true_clf_5c)
        y_pred_clf_3c = class_5to3(y_pred_clf_5c)
        F1_micro_3c = f1_score(y_true_clf_3c, y_pred_clf_3c, average='micro')
        F1_micro_3c = np.round(F1_micro_3c, decimals=4)
        return F1_micro_5c, F1_micro_3c
    elif mode == 'DG_R':
        # 5 class
        y_true_clf_5c = ArR_DDG_dataset.data.y[:,1].numpy()
        y_pred_reg, y_pred_clf_5c = DG_to_DDG(idx, y_pred)
        F1_micro_5c = f1_score(y_true_clf_5c, y_pred_clf_5c, average='micro')
        F1_micro_5c = np.round(F1_micro_5c, decimals=4)
        # 3 class
        y_true_clf_3c = class_5to3(y_true_clf_5c)
        y_pred_clf_3c = class_5to3(y_pred_clf_5c)
        F1_micro_3c = f1_score(y_true_clf_3c, y_pred_clf_3c, average='micro')
        F1_micro_3c = np.round(F1_micro_3c, decimals=4)
        return F1_micro_5c, F1_micro_3c
    else:
        y_true_clf_5c, y_pred_clf_5c = y_true, y_pred
        y_true_clf_3c = class_5to3(y_true_clf_5c)
        y_pred_clf_3c = class_5to3(y_pred_clf_5c)
        F1_micro_3c = f1_score(y_true_clf_3c, y_pred_clf_3c, average='micro')
        F1_micro_3c = np.round(F1_micro_3c, decimals=4)
        return F1_micro_3c
