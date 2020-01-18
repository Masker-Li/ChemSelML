from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, auc, f1_score, roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.utils import shuffle
import torch
from itertools import combinations
import time
import numpy as np
from bidict import bidict
from joblib import Parallel, delayed

from ..learners.Regressor import get_regressor_lib
from ..learners.Classifier import get_classifier_lib
from ..train.reg2clf import reg2clf_Evaluation

global_feats_lib = ['mergefp', 'BoB', 'PhyChem_total', ] # , 'mergefp'， 'MACCSfp', 'Morganfp'
local_feats_lib = ['SOAP', 'ACSF_local', 'PhyChem_local', ]

reg_lib = get_regressor_lib()
clf_lib = get_classifier_lib()
n_jobs = 2

def get_data(dataset, features_type, mode='DDG_R', random_state=1024):
    '''
    dataset: ArR_dataset.data
    features_type: str or list
    '''
    assert type(features_type) in [list, str]
    assert mode in ['DG_R', 'DDG_R',
                    'DDG_C'], 'mode should in DG_R/DDG_R/DDG_C'
    if type(features_type) == str:
        features_type = [features_type]
    f_set = []
    if mode == 'DG_R':
        for f in features_type:
            f_set.append(dataset['Ar_'+f].float())
            f_set.append(dataset['R_'+f].float())
        X = torch.cat(f_set, dim=1).numpy()
        y = dataset.y.numpy()

        idx = dataset.ArR_idx
        idx, X, y = shuffle(idx, X, y, random_state=random_state)
        return idx, X, y
    else:
        local_features_lib = ['SOAP', 'ACSF_local', 'PhyChem_local', ]

        for f in features_type:
            if f in local_features_lib:
                f_set += [dataset['Ar_'+f+'@A'].float(),
                          dataset['Ar_'+f+'@B'].float()]
                f_set += [dataset['R_'+f].float()]
            else:
                f_set.append(dataset['Ar_'+f].float())
                f_set.append(dataset['R_'+f].float())
        X = torch.cat(f_set, dim=1).numpy()
        if mode == 'DDG_R':
            y = dataset.y[:, 0].numpy()
        elif mode == 'DDG_C':
            y = dataset.y[:, 1].int().numpy()

        idx = dataset.ArR_sel_idx
        idx, X, y = shuffle(idx, X, y, random_state=random_state)
        return idx, X, y


def MultiLables_AUC_micro(y_true, y_pred_proba, classes):
    y_pred_proba2 = y_pred_prob2

    y_bin = label_binarize(y_true, classes=np.unique(y_true))
    n_classes = y_bin.shape[1]
    fpr, tpr, roc_auc = dict(), dict(), dict()

    #y_pred_proba = selector.predict_proba(X)
    for i, class_name in zip(range(n_classes), classes):
        fpr[class_name], tpr[class_name], thresholds = roc_curve(
            y_bin[:, i], y_pred_proba[:, i])
        roc_auc[class_name] = auc(fpr[class_name], tpr[class_name])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_bin.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return roc_auc["micro"]


def get_CV_res(model, X, y, mode='DDG_R'):
    assert mode in ['DG_R', 'DDG_R',
                    'DDG_C'], 'mode should in DG_R/DDG_R/DDG_C'
    cv_results = dict()
    if mode in ['DG_R', 'DDG_R']:
        y_pred = cross_val_predict(model, X, y, cv=5)
        R2 = r2_score(y, y_pred)
        MAE = mean_absolute_error(y, y_pred)
        MSE = mean_squared_error(y, y_pred)

        cv_results['R2'] = np.round(R2, decimals=4)
        cv_results['MAE'] = np.round(MAE, decimals=4)
        cv_results['MSE'] = np.round(MSE, decimals=4)

    elif mode == 'DDG_C':
        y_pred_proba = cross_val_predict(
            model, X, y, cv=5, method='predict_proba')
        classes_ = np.array([-2, -1,  0,  1,  2], dtype=np.int32)
        y_pred = classes_.take(np.argmax(y_pred_proba, axis=1), axis=0)
        CA = accuracy_score(y, y_pred)
        ROC_AUC_micro = MultiLables_AUC_micro(y, y_pred_proba, classes_)
        F1_micro = f1_score(y, y_pred, average='micro')

        cv_results['CA'] = np.round(CA, decimals=4)
        cv_results['ROC_AUC_micro'] = np.round(ROC_AUC_micro, decimals=4)
        cv_results['F1_micro_5c'] = np.round(F1_micro, decimals=4)

    return y_pred, cv_results


class FeatureSet:
    def __init__(self, global_feats_lib=None, local_feats_lib=None):
        self.feat_comb_prepare(global_feats_lib, local_feats_lib)
        self.alias_dict = {'mergefp': 'F', 'BoB': 'B', 'PhyChem_total': 'P',
                           'SOAP': 's', 'ACSF_local': 'a', 'PhyChem_local': 'p'}
        self.alias_dict_inverse = bidict(self.alias_dict).inverse

    def feat_comb_prepare(self, global_feats_lib=None, local_feats_lib=None):
        self.global_feats_lib = global_feats_lib
        self.local_feats_lib = local_feats_lib
        if not global_feats_lib:
            self.global_feats_lib = ['mergefp', 'BoB', 'PhyChem_total', ]  # , 'mergefp'， 'MACCSfp', 'Morganfp'
        if not local_feats_lib:
            self.local_feats_lib = ['SOAP', 'ACSF_local', 'PhyChem_local', ]

        global_feats_set = []
        for i in range(0, len(global_feats_lib)+1):
            global_feats_set += list(combinations(global_feats_lib, i))
        local_features_set = []
        for i in range(1, len(local_feats_lib)+1):
            local_features_set += list(combinations(local_feats_lib, i))
        self.feats_set = []
        for global_feats in global_feats_set:
            for local_feats in local_features_set:
                self.feats_set.append(global_feats + local_feats)

    def feat2alias(self, feat):
        '''feat: str or list'''
        assert type(feat) in [str, list, tuple]

        if type(feat) == str:
            return self.alias_dict[feat]
        else:
            tmp = [self.alias_dict[x] for x in feat]
            tmp = sorted(tmp)
            return ''.join(tmp)

    def alias2feat(self, alias):
        '''alias: str'''
        assert type(alias) == str

        if len(alias) == 1:
            return tuple([self.alias_dict_inverse[alias]])
        else:
            return tuple([self.alias_dict_inverse[x] for x in alias])

    def get_feat_idx(self, feat):
        return self.feats_set.index(feat)

    def get_feat_from_idx(self, idx):
        return self.feats_set[idx]
    
    def FeatSet2AliasSet(self, feats_set=None):
        if not feats_set: feats_set = self.feats_set
        return [self.feat2alias(x) for x in feats_set]
    
    def AliasSet2FeatSet(self, alias_set=[]):
        assert type(alias_set) in [str, list], 'alias_set shoule be None, a str or a list'
        if type(alias_set)==str: alias_set = [alias_set]
        return [FSet.alias2feat(x) for x in alias_set]

FSet = FeatureSet(global_feats_lib=global_feats_lib, local_feats_lib=local_feats_lib)


def feat_comb_prepare(feat_alias=None, feats_lib=None, FSet=FSet):
    '''
        Parameters
        ----------
        feat_alias: string or list of string, string should contains only the following letters: 
                    'F', 'B', 'P', 's', 's', 'p'. Its meaning is：
                    {'F': 'mergefp', 'B': 'BoB', 'P': 'PhyChem_total', 
                     's': 'SOAP', 'a': 'ACSF_local', 'p': 'PhyChem_local'}
        
        feats_lib: base feature type used to combinations, if feat_combs not None, feats_lib will be ignored.
    '''
    global_feats_lib = ['mergefp', 'BoB', 'PhyChem_total', ] # , 'mergefp'， 'MACCSfp', 'Morganfp'
    local_feats_lib = ['SOAP', 'ACSF_local', 'PhyChem_local', ]
    
    if feat_alias:
        assert type(feat_alias) in [str, list], 'feat_alias shoule be None, a str or a list'
        if type(feat_alias)==str: feat_alias = [feat_alias]
        feat_combs = FSet.AliasSet2FeatSet(feat_alias)
    else:
        global_feat, local_feat = [], []
        if feats_lib:
            for x in feats_lib:
                if x in global_feats_lib: global_feat.append(x)
                if x in local_feats_lib: local_feat.append(x)
        else:
            global_feat = global_feats_lib
            local_feat = local_feats_lib
        
        FSet.feat_comb_prepare(global_feat, local_feat)
        feat_combs = FSet.feats_set
    
    return feat_combs, FSet


def get_models(model_names=['RF_R'], mode='DDG_R', Extra_model_dict=None, clf_lib=clf_lib, reg_lib=reg_lib, n_jobs=n_jobs):
    '''
        Parameters
        ----------
        model_names:  list, eg: ['RF_R','XGB_R']
        
        Extra_model_dict: {k, model} like.
        
        
        generate:
        -----------------
        model_lib: all model lib
        model_dict: used model
    '''
    if Extra_model_dict:
        if mode=='DDG_C': 
            clf_lib = {**clf_lib, **Extra_model_dict}
        elif mode in ['DG_R', 'DDG_R']:
            reg_lib = {**reg_lib, **Extra_model_dict}
        model_names += list(Extra_model_dict.keys())
    
    model_names = list(set(model_names))
    model_lib = clf_lib if mode=='DDG_C' else reg_lib
    if model_names:
        model_dict = dict()
        for k in model_names:
            _model = model_lib[k]
            if hasattr(_model, 'n_jobs'):
                _model.n_jobs = n_jobs
            model_dict[k] = _model
    else:
        model_dict = model_lib
    
    return model_lib, model_dict


def _CV_reg_fit(idx, X, y, reg_name, reg, feat_idx, mode='DDG_R', FSet=FSet, reg_lib=reg_lib):
    '''column for reg_summary: ['Feat_idx', 'Feat_alias', 'Model', 'MAE', 'MSE', 'R2', 'F1_micro_5c', 'F1_micro_3c']'''
    assert mode in ['DG_R', 'DDG_R'], 'mode should in DG_R/DDG_R'
    
    feat_type = FSet.feats_set[feat_idx]
    feat_name = FSet.feat2alias(feat_type)
    
    y_pred, reg_res = get_CV_res(reg, X, y, idx, mode)
    F1_micro_5c, F1_micro_3c = reg2clf_Evaluation(y, y_pred, mode=mode)
    reg_res['Features'] = feat_type
    reg_res['Features_name'] = feat_name
    reg_res['Regressor'] = reg_name
    reg_res['F1_micro_5c'] = np.round(F1_micro_5c, decimals=4)
    reg_res['F1_micro_3c'] = np.round(F1_micro_3c, decimals=4)
    reg_summary = [feat_idx, feat_name, reg_name, reg_res['MAE'], reg_res['MSE'], reg_res['R2'],
                        reg_res['F1_micro_5c'], reg_res['F1_micro_3c']]

    return reg_summary, y_pred, reg_res


def _CV_clf_fit(idx, X, y, clf_name, clf, feat_idx, mode='DDG_R', FSet=FSet, clf_lib=clf_lib):
    '''column for clf_summary: ['Feat_idx', 'Feat_alias', 'Model', 'CA', 'ROC_AUC_micro', 'F1_micro_5c', 'F1_micro_3c']'''
    feat_type = FSet.feats_set[feat_idx]
    feat_name = FSet.feat2alias(feat_type)
    
    y_pred, clf_res = get_CV_res(clf, X, y, idx, mode)
    F1_micro_3c = reg2clf_Evaluation(y, y_pred, mode=mode)
    clf_res['Features'] = feat_type
    clf_res['Features_name'] = feat_name
    clf_res['Cassifier'] = clf_name
    clf_res['F1_micro_3c'] = np.round(F1_micro_3c, decimals=4)
    clf_summary = [feat_idx, feat_name, clf_name, clf_res['CA'], clf_res['ROC_AUC_micro'],
                        clf_res['F1_micro_5c'], clf_res['F1_micro_3c']]
    return clf_summary, y_pred, clf_res


def _CV_fit(dataset, model_dict=None, mode='DDG_C', feat_type=None, FSet=FSet, model_lib=None, n_jobs=n_jobs, parallel=False):
    assert mode in ['DG_R', 'DDG_R',
                    'DDG_C'], 'mode should in DG_R/DDG_R/DDG_C'
    
    start_time = time.time()
    
    CV_model_fit = _CV_clf_fit if mode=='DDG_C' else _CV_reg_fit
    if not model_lib:
        model_lib = clf_lib if mode=='DDG_C' else reg_lib
    idx, X, y = get_data(dataset, list(feat_type), mode, random_state=1024)
    model_res = dict()
    feat_history = dict()
    y_pred_dict = {'idx': idx, 'y': y}
    feat_idx = FSet.get_feat_idx(feat_type)
    if n_jobs:
        n_jobs = min(n_jobs, len(model_dict))
    if parallel:
        res = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(CV_model_fit)(idx, X, y, m_name, model, feat_idx, mode, 
                                      FSet, model_lib) for m_name, model in model_dict.items())
        
        for i, m_name in enumerate(model_dict.keys()):
            j = list(model_lib.keys()).index(m_name)
            summary, y_pred, tmp_res = res[i]
            model_res[feat_idx*100+j] = summary
            
            y_pred_dict[m_name] = y_pred
            feat_history[m_name] = tmp_res
    else:
        for i, (m_name, model) in enumerate(model_dict.items()):
            j = list(model_lib.keys()).index(m_name)
            summary, y_pred, tmp_res = CV_model_fit(X, y, m_name, model, feat_idx, mode, FSet, model_lib)
            model_res[feat_idx*100+j] = summary
            
            y_pred_dict[m_name] = y_pred
            feat_history[m_name] = tmp_res
            
    end_time = time.time()
    Timestamp = time.strftime(
        '%Y.%m.%d %H:%M:%S', time.localtime(end_time))
    print(feat_idx, Timestamp, feat_type, 'complete!')
    print('    Time cost: %.4fs' %(end_time-start_time))
    return model_res, y_pred_dict, feat_history


feat_args = {'feat_alias': None, 'feats_lib': None}
model_args = {'model_names': [], 'Extra_model_dict': None, 'clf_lib': clf_lib, 'reg_lib': reg_lib}

def CV_fit(dataset, mode='DDG_R', features_set=None, model_lib=None, model_dict=None, n_jobs=n_jobs, FSet=FSet, 
           feat_parallel=False, model_parallel=False):
    '''Extra_model_dict: {k, model} like. '''
    assert mode in ['DG_R', 'DDG_R',
                    'DDG_C'], 'mode should in DG_R/DDG_R/DDG_C'
    if not features_set:
        features_set, FSet= feat_comb_prepare(FSet=FSet, **feat_args)
    if not model_lib:
        model_lib = clf_lib if mode=='DDG_C' else reg_lib
    if not model_dict:
        model_lib, model_dict = get_models(mode=mode, **model_args)
    if type(features_set[0]) == str:
        features_set = [features_set]
    
    cv_result_history = dict()
    model_res = dict()
    y_pred_history = dict()
    
    kwargs = {'dataset': dataset, 'model_dict': model_dict, 'mode': mode, 'FSet': FSet, 
           'model_lib': model_lib, 'n_jobs': n_jobs, 'parallel': model_parallel}
    
    feat_idxs = np.arange(len(features_set))
    if n_jobs:
        n_jobs = min(n_jobs, len(feat_idxs)*len(model_dict))
    if feat_parallel:
        res = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_CV_fit)(feat_type=features_set[i], **kwargs) for i in feat_idxs)
        
        for i in feat_idxs:
            m_res, y_pred_dict, feat_hist = res[i]
            model_res = {**model_res, **m_res}
            y_pred_history[i] = y_pred_dict
            cv_result_history[i] = feat_hist
    else:
        for i in feat_idxs:
            m_res, y_pred_dict, feat_hist = _CV_fit(feat_type=features_set[i], **kwargs)
            model_res = {**model_res, **m_res}
            y_pred_history[i] = y_pred_dict
            cv_result_history[i] = feat_hist
    
    if mode == 'DDG_C':
        column = ['Feat_idx', 'Feat_alias', 'Model', 'CA', 
                  'ROC_AUC_micro', 'F1_micro_5c', 'F1_micro_3c']
        res_df = pd.DataFrame.from_dict(model_res, orient='index', 
                        columns=column).sort_values(by=['F1_micro_5c'], ascending=False)
    else:
        column = ['Feat_idx', 'Feat_alias', 'Model', 'MAE',
                  'MSE', 'R2', 'F1_micro_5c', 'F1_micro_3c']
        res_df = pd.DataFrame.from_dict(model_res, orient='index', 
                    columns=column).sort_values(by=['R2'], ascending=False)
    return res_df, y_pred_history, FSet, cv_result_history

