import os 
import sys
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
from sklearn.externals import joblib
import pickle
from ..train.training import get_data
from ..train.training import get_CV_res
from ..train.utils import get_RFECV_result, get_pred_data, actual_vs_pred
from ..train.utils import Plot_RFECV, Barh_Feature_Ranking, Plot_True_vs_Pred
from ..train.utils import plot_learning_curve, get_accurancy


class ChemSel_Predictor:
    def __init__(self, dataset, model, features, mode='DDG_R', n_jobs=None,
             processed_dir=None, suffix=None, reloadTimestamp=None):
        assert mode in ['DG_R', 'DDG_R',
                        'DDG_C'], 'mode should in DG_R/DDG_R/DDG_C'
        self.dataset = dataset
        self.model = model
        self.features = features
        self.mode = mode
        self.n_jobs = n_jobs
        self.processed_dir = processed_dir
        self.suffix = suffix
        self.reloadTimestamp = reloadTimestamp

        if hasattr(self.model, 'n_jobs'):
            self.model.n_jobs = n_jobs
        self.idx, self.X, self.y = get_data(
            self.dataset, self.features, mode=self.mode)
        print('X.shape: ', self.X.shape)
        # Data standardization
        self.scaler = StandardScaler()
        self.scaler.fit(self.X)
        self.LoadDataTime = time.strftime(
            '%Y%m%d_%H%M%S', time.localtime(time.time()))
        print(self.LoadDataTime)
        self.CrossValidation()
        print('Please input self.RFECV_Train() to start training model and get self.selector')

    def Load_test_data(self, test_dataset):
        # import ExtraSet data and random
        tidx, tX, ty = get_data(test_dataset, self.features, mode=self.mode)
        self.tidx, self.tX, self.ty = tidx, self.scaler.transform(tX), ty
        

    def CrossValidation(self):
        self.X = self.scaler.transform(self.X)
        y_pred, self.cv_results = get_CV_res(
            self.model, self.X, self.y, mode=self.mode)
        print('Cross Validation:')
        for k, v in self.cv_results.items():
            print('    **',k,' :', np.round(v.mean(), 4))
        

    # Start RFECV
    def RFECV_Train(self):
        model_folder = 'models_pkg'
        if os.path.isdir(r"%s/%s" % (self.processed_dir, model_folder)) == False:
            os.makedirs(r"%s/%s" % (self.processed_dir, model_folder))

        if self.reloadTimestamp == None:
            #loc1 = np.where(y==y.min())
            #loc2 = np.where(y==y.max())
            #X2 = np.delete(X, [loc1, loc2], axis=0)
            #y2 = np.delete(y, [loc1, loc2])

            self.selector = get_RFECV_result(self.model, self.X, self.y, self.n_jobs)
            save_path = r'%s/%s/FinalModel_%s_%s.pkl'% (self.processed_dir, 
                                        model_folder, self.suffix, self.LoadDataTime)
            joblib.dump(self.selector, save_path)            
        else:
            save_path = r'%s/%s/FinalModel_%s_%s.pkl'%(self.processed_dir, 
                                        model_folder, self.suffix, self.reloadTimestamp)
            self.selector = joblib.load(save_path)
            self.RFECV_f1_score = self.selector.grid_scores_[
                self.selector.grid_scores_.argmax()]
            self.LoadDataTime = self.reloadTimestamp
        print(save_path)

    def _get_learning_curve(self, notitle=False):
        title = "Learning Curves (%s)" % self.suffix if not notitle else None
        # Cross validation with 100 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a validation set.
        cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(self.model, self.X, self.y, cv=cv, n_jobs=12,
                           train_sizes=np.linspace(.1, 1.0, 20),
                           return_times=True)
        self.LearnCurve_kwargs = {'train_sizes': train_sizes, 'train_scores': train_scores,
                                  'test_scores': test_scores, 'fit_times': fit_times,
                                  'title': title, 'ylim': (0.7, 1.01)}

    def _get_prediction(self, idx, X, y):
        idx, y_true, y_pred, Models = get_pred_data(self.selector, idx, X, y)

        pred_dict = {'idx': idx, 'y_true': y_true, 'y_pred': y_pred}
        pred_df = pd.DataFrame.from_dict(pred_dict)
        columns = list(pred_df.columns)
        # S R Ar loc1 loc2 for X XX XXX X X
        pred_df['R'] = pred_df['idx']//100000 % 100
        # S R Ar loc1 loc2 for X XX XXX X X
        pred_df['Ar'] = pred_df['idx'] % 100000//100
        # S R Ar loc1 loc2 for X XX XXX X X
        pred_df['loc1'] = pred_df['idx'] % 100//10
        # S R Ar loc1 loc2 for X XX XXX X X
        pred_df['loc2'] = pred_df['idx'] % 10
        columns = columns[0:1] + ['Ar', 'R', 'loc1', 'loc2'] + columns[1:]
        pred_df = pred_df[columns]
        pred_df = pred_df.astype({'Ar': 'int32', 'R': 'int32', 'loc1': 'int32', 'loc2': 'int32'})
        pred_df.set_index('idx', inplace=True)

        ArR_df, ArR_df_sel = actual_vs_pred(pred_df, neg_DDG_cutoff=1.42)
        ArR_df_sel, site_acc, degree_acc = get_accurancy(ArR_df_sel)
        res_dict = {'idx': idx, 'y_true': y_true, 'y_pred': y_pred,
                    'pred_df': pred_df, 'ArR_df': ArR_df, 'ArR_df_sel': ArR_df_sel,
                    'site_acc': site_acc, 'degree_acc': degree_acc}
        return res_dict

    # Plot number of features VS. cross-validation scores
    def get_training_result(self, storage_folder='TrainSet_result', notitle=False):
        if os.path.isdir(r"%s/%s" % (self.processed_dir, storage_folder)) == False:
            os.makedirs(r"%s/%s" % (self.processed_dir, storage_folder))

        if not hasattr(self, 'LearnCurve_kwargs'):
            self._get_learning_curve(notitle=notitle)
        plot_learning_curve(**self.LearnCurve_kwargs, figure_file=r'%s/%s/LearningCurve_%s_%s.png' % (
            self.processed_dir, storage_folder, self.suffix, self.LoadDataTime))

        Plot_RFECV(self.selector, figure_file=r'%s/%s/RFECV_FeatureSelection_%s_%s.png' % (
            self.processed_dir, storage_folder, self.suffix, self.LoadDataTime))

        Barh_Feature_Ranking(self.selector, best_k=15,
                             figure_file=r'%s/%s/RFECV_FeatureRanking_%s_%s_best15.png' % (
                                 self.processed_dir, storage_folder, self.suffix, self.LoadDataTime))
        # all feature ranking
        Barh_Feature_Ranking(self.selector, best_k=None,
                             figure_file=r'%s/%s/RFECV_FeatureRanking_%s_%s_all.png' % (
                                 self.processed_dir, storage_folder, self.suffix, self.LoadDataTime))

        if not hasattr(self, 'pred_df'):
            res_dict = self._get_prediction(self.idx, self.X, self.y)
            for k, value in res_dict.items():
                self.__dict__[k] = value
            self.pred_df.to_csv(r'%s/%s/TrainSet_DDG_Pred_site_vs_site_%s_%s.csv' % (
                self.processed_dir, storage_folder, self.suffix, self.LoadDataTime))
            self.ArR_df.to_csv(r'%s/%s/TrainSet_DDG_Pred_ArR_site_sort_%s_%s.csv' % (
                self.processed_dir, storage_folder, self.suffix, self.LoadDataTime))
            self.ArR_df_sel.to_csv(r'%s/%s/TrainSet_selection_Pred_ArR_site_sort_%s_%s.csv' % (
                self.processed_dir, storage_folder, self.suffix, self.LoadDataTime))

        title = self.suffix if not notitle else None
        Plot_True_vs_Pred(self.y_true, self.y_pred, title=title,
                          figure_file=r'%s/%s/True_vs_Pred_%s_%s.png' % (
                              self.processed_dir, storage_folder, self.suffix, self.LoadDataTime))
        print('site_acc: ', self.site_acc)
        print('degree_acc: ', self.degree_acc)
        

    def get_test_result(self, test_dataset=None, suffix='TestSet', storage_folder='TestSet_result', notitle=False):
        if os.path.isdir(r"%s/%s" % (self.processed_dir, storage_folder)) == False:
            os.makedirs(r"%s/%s" % (self.processed_dir, storage_folder))

        try:
            self.Load_test_data(test_dataset)
            tidx, X, y = self.tidx, self.tX, self.ty
        except:
            print(r'Wrong: test_dataset Mismatched, please input a test_dataset!')

        if not hasattr(self, 'test_pred'):
            self.test_pred = self._get_prediction(tidx, X, y)
            self.test_pred['pred_df'].to_csv(r'%s/%s/TestSet_DDG_Pred_site_vs_site_%s_%s_%s.csv' % (
                self.processed_dir, storage_folder, suffix, self.suffix, self.LoadDataTime))
            self.test_pred['ArR_df'].to_csv(r'%s/%s/TestSet_DDG_Pred_ArR_site_sort_%s_%s_%s.csv' % (
                self.processed_dir, storage_folder, suffix, self.suffix, self.LoadDataTime))
            self.test_pred['ArR_df_sel'].to_csv(r'%s/%s/TestSet_selection_Pred_ArR_site_sort_%s_%s_%s.csv' % (
                self.processed_dir, storage_folder, suffix, self.suffix, self.LoadDataTime))

        title = '%s_%s'%(suffix, self.suffix) if not notitle else None
        Plot_True_vs_Pred(self.test_pred['y_true'], self.test_pred['y_pred'], title=title,
                          figure_file=r'%s/%s/True_vs_Pred_%s_%s.png' % (
            self.processed_dir, storage_folder, self.suffix, self.LoadDataTime))
        print('test_site_acc: ', self.test_pred['site_acc'])
        print('test_degree_acc: ', self.test_pred['degree_acc'])
        
    def save_to_pkl(self, Predictor, filename='Predictor', storage_folder='models_pkg'):
        if os.path.isdir(r"%s/%s" % (self.processed_dir, storage_folder)) == False:
            os.makedirs(r"%s/%s" % (self.processed_dir, storage_folder))

        fn = r'%s/%s/%s_%s_%s.pkl' % (self.processed_dir, storage_folder, 
                                             filename, self.suffix, self.LoadDataTime)
        print(fn)
        with open(fn,'wb') as f:
            pickle.dump(Predictor,f,0)  
    
    def load_from_pkl(self, filename='Predictor', storage_folder='models_pkg'):
        if os.path.isdir(r"%s/%s" % (self.processed_dir, storage_folder)) == False:
            os.makedirs(r"%s/%s" % (self.processed_dir, storage_folder))

        if filename[-4:] == '.pkl':
            fn = filename
        else:
            fn = r'%s/%s/%s_%s_%s.pkl' % (self.processed_dir, storage_folder, 
                                             filename, self.suffix, self.reloadTimestamp)
        print(fn)
        with open(fn,'rb') as f:
            Predictor = pickle.load(f)  
        return Predictor