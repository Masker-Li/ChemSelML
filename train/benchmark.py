import time
import numpy as np
import pandas as pd
import os, sys
import pickle

from ..train.training import FSet, CV_fit, get_models, feat_comb_prepare
from ..train.training import reg_lib, clf_lib


class Benchmark:
    def __init__(self, mode='DDG_R', model_names=['RF_R'], Extra_model_dict=None, \
                 feat_combs=None, feat_alias=None, feats_lib=None, \
                 n_jobs=None, processed_dir=None, suffix=None, reloadTimestamp=None, \
                 FSet=FSet, clf_lib=clf_lib, reg_lib=reg_lib, feat_parallel=False, model_parallel=False):
        
        '''
            Parameters
            ----------
            mode: values of {'DG_R', 'DG_R', 'DDG_C'}, 
                    DG_R means regression of DG, 
                    DDG_R means regression of DDG,
                    DDG_C means classification of DDG
            
            model_names:  list, eg: ['RF_R','XGB_R']
            
            Extra_model_dict: {k, model} like.
            
            feat_combs: list of feature combinations, if None, use all possible combinations.
            
            feat_alias: string or list of string, string should contains only the following letters: 
                        'F', 'B', 'P', 's', 's', 'p'. Its meaning is：
                        {'F': 'mergefp', 'B': 'BoB', 'P': 'PhyChem_total', 
                         's': 'SOAP', 'a': 'ACSF_local', 'p': 'PhyChem_local'}
            
            feats_lib: base feature type used to combinations, if feat_combs not None, feats_lib will be ignored.
            
            n_jobs: int or None, optional (default=None). The number of jobs to run in parallel.
            
            processed_dir: directory used to save generated files
            
            suffix: specified suffix of filename
            
            reloadTimestamp: used to reload saved files
        '''
        assert mode in ['DG_R', 'DDG_R',
                    'DDG_C'], 'mode should in DG_R/DDG_R/DDG_C'
        assert not (feat_alias and feat_combs), '\
            feat_combs and feat_alias are mutually exclusive and can \
            accept at most one of the parameters'
        
        self.mode = mode
        self.n_jobs = n_jobs
        self.processed_dir = processed_dir
        self.suffix = suffix
        self.reloadTimestamp = reloadTimestamp
        self.feat_parallel = feat_parallel
        self.model_parallel = model_parallel
        if not reloadTimestamp:
            self.LoadDataTime = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        else:
            self.LoadDataTime = reloadTimestamp
            
        if feat_combs:
            self.feat_combs, self.FSet = feat_combs, FSet  
        else:
            self.feat_comb_prepare(feat_alias, feats_lib, FSet)
        self.get_models(model_names, Extra_model_dict, clf_lib, reg_lib)
        
    def feat_comb_prepare(self, feat_alias=None, feats_lib=None, FSet=FSet):
        '''
            Parameters
            ----------
            feat_combs: list of feature combinations, if None, use all possible combinations.
            
            feat_alias: string or list of string, string should contains only the following letters: 
                        'F', 'B', 'P', 's', 's', 'p'. Its meaning is：
                        {'F': 'mergefp', 'B': 'BoB', 'P': 'PhyChem_total', 
                         's': 'SOAP', 'a': 'ACSF_local', 'p': 'PhyChem_local'}
            
            feats_lib: base feature type used to combinations, if feat_combs not None, feats_lib will be ignored.
        '''
        global_feats_lib = ['mergefp', 'BoB', 'PhyChem_total', ] # , 'mergefp'， 'MACCSfp', 'Morganfp'
        local_feats_lib = ['SOAP', 'ACSF_local', 'PhyChem_local', ]
        
        self.feat_alias = feat_alias
        self.feats_lib = feats_lib
        self.feat_combs, self.FSet = feat_comb_prepare(feat_alias, feats_lib, FSet)
    
    def get_models(self, model_names=['RF_R'], Extra_model_dict=None, clf_lib=clf_lib, reg_lib=reg_lib):
        '''
            Parameters
            ----------
            model_names:  list, eg: ['RF_R','XGB_R']
            
            Extra_model_dict: {k, model} like.
            
            
            generate:
            -----------------
            self.model_lib: all model lib
            self.model_dict: used model
        '''
        self.Extra_model_dict = Extra_model_dict
        self.model_lib, self.model_dict = get_models(model_names, 
                 self.mode, Extra_model_dict, clf_lib, reg_lib, n_jobs=self.n_jobs)
        self.model_names = list(self.model_dict.keys())

    
    def CV_fit(self, dataset, feat_combs=None, model_dict=None, inplace=False, **kwargs):
        '''
            Parameters
            ----------
            dataset:  dataset of torch_geometric.data.Data like, eg: ReactionDataset.data or SelectivityDataset.data
            
            feat_combs: list of feature combinations, if None, use all possible combinations.
            
            model_dict: {k, model} like.
            
            default values for kwargs:
                {'dataset': self.dataset, 'mode': self.mode, 
                 'features_set': feat_combs, 'model_lib': self.model_lib, 
                 'model_dict': model_dict, 'n_jobs': self.n_jobs, 'FSet': self.FSet}
        '''
        feat_combs = feat_combs if feat_combs else self.feat_combs
        model_dict = model_dict if model_dict else self.model_dict
        _kwargs = {'dataset': dataset, 'mode': self.mode, 
                   'features_set': feat_combs, 'model_lib': self.model_lib, 
                   'model_dict': model_dict, 'n_jobs': self.n_jobs, 'FSet': self.FSet, 
                   'feat_parallel': self.feat_parallel, 'model_parallel': self.model_parallel}
        if kwargs:
            for k, v in kwargs:
                _kwargs[k] = v
        inplace = True if not (feat_combs or model_dict) else inplace
        
        model_res, y_pred_history, FSet, cv_result_history = CV_fit(**_kwargs)
        if inplace:
            self.summary = model_res
            self.y_pred_history = y_pred_history
            self.FSet = FSet
            self.CV_history = cv_result_history
        return model_res, y_pred_history, FSet, cv_result_history
    
    def save_to_pkl(self, obj, filename='Benchmark', storage_folder='Benchmark_pkl'):
        if os.path.isdir(r"%s/%s" % (self.processed_dir, storage_folder)) == False:
            os.makedirs(r"%s/%s" % (self.processed_dir, storage_folder))

        fn = r'%s/%s/%s_%s_%s.pkl' % (self.processed_dir, storage_folder, 
                                             filename, self.suffix, self.LoadDataTime)
        print(fn)
        with open(fn,'wb') as f:
            pickle.dump(obj,f,0)  
        
    def load_from_pkl(self, filename='Benchmark', storage_folder='Benchmark_pkl'):
        if os.path.isdir(r"%s/%s" % (self.processed_dir, storage_folder)) == False:
            os.makedirs(r"%s/%s" % (self.processed_dir, storage_folder))

        fn = r'%s/%s/%s_%s_%s.pkl' % (self.processed_dir, storage_folder, 
                                             filename, self.suffix, self.reloadTimestamp)
        print(fn)
        with open(fn,'rb') as f:
            obj = pickle.load(f)  
        return obj
    
    def __repr__(self):
        return '{}(n_feat_combs={}, n_models={})'.format(self.__class__.__name__,
                                        len(self.feat_combs), len(self.model_dict))
