import math
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.feature_selection import RFECV
import sklearn as skl
from sklearn.model_selection import StratifiedKFold, KFold
'''KFold for Regression and StratifiedKFold for Classification'''

from ..bin.featurization import PhyChem
from ..bin.featurization import PhyChem_transform as PCT


def get_PhyChem_features_name(suffix='dev'):
    Ar_ = PhyChem(mode='Ar', suffix=suffix)
    R_ = PhyChem(mode='R', suffix=suffix)
    features = [r'%s$_{\rm Ar}$'%x for x in PCT(Ar_.total_PhyChem_labels[1:])] + \
        [r'%s$_{\rm Ar}$@A'%x for x in PCT(Ar_.local_PhyChem_labels[1:])] + \
        [r'%s$_{\rm Ar}$@B'%x for x in PCT(Ar_.local_PhyChem_labels[1:])] + \
        [r'%s$_{\rm R}$'%x[1:] for x in PCT(R_.total_PhyChem_labels[1:])] +  \
        [r'%s$_{\rm R}$'%x[1:] for x in PCT(R_.local_PhyChem_labels[1:])] 
    return features


def Plot_RFECV(selector, figure_file=None):
    sns.set(style='whitegrid', palette='muted', color_codes=True)
    #sns.set(style="ticks")
    #plt.rcParams['mathtext.fontset'] = 'custom'
    #plt.rcParams['mathtext.it'] = 'Arial:italic'
    #plt.rcParams['mathtext.bf'] = 'Arial:italic:bold'

    # Plot number of features VS. cross-validation scores
    min_f = selector.min_features_to_select
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.xlabel("Feature Number", fontsize=24)
    plt.ylabel("Cross Validation Score (R$^2$)", fontsize=24)
    x_max = len(selector.grid_scores_) + min_f
    plt.plot(range(min_f, x_max),
             selector.grid_scores_)
    x_index = selector.grid_scores_.argmax()
    x_min2,x_max2 = ax.get_xlim()
    y_max = selector.grid_scores_[x_index]
    y_min = selector.grid_scores_.min()
    y_min2 = y_min//0.01*0.01-0.002
    y_max2 = y_max//0.01*0.01+0.01+0.003
    plt.ylim((y_min2, y_max2))
    # plt.scatter(x_index+min_f,y_max,s=25,color='',marker='o',edgecolors='darkgreen',lw=1.2)
    plt.axhline(y=y_max, xmin=0, xmax=(x_index-x_min2)/(x_max-x_min2), c='g', linestyle='--',
                lw=1, label=r'y = {0}'''.format(y_max))
    plt.axvline(x=x_index+min_f, ymin=0, ymax=(y_max-y_min2)/(y_max2-y_min2), c='g',
                linestyle='--', lw=1, label=r'x = {0}'''.format(x_index+min_f))
    plt.text(4*x_max/9, y_min+(y_max-y_min)/9, 'Max R$^2$ Score: {0:0.3f}\nFeature Number: {1}'''.format(
        y_max, x_index+min_f), fontdict={'size': '20', 'color': 'black'})
    if figure_file != None:
        plt.savefig(figure_file, dpi=300)
    plt.show()

    print('Feature number: ',
          selector.estimator_.feature_importances_.shape[0])
    print('Max_R2_Score: ', selector.grid_scores_.max())


def get_pred_data(selector, idx, X, y):
    cv_predictor = clone(selector.estimator_)
    y_idx = np.empty(0)
    y_true = np.empty(0)
    y_pred = np.empty(0)
    Models = dict()
    for (train, test), i in zip(selector.cv.split(idx, X, y), range(selector.cv.get_n_splits())):
        # print(test)
        cv_predictor.fit(selector.transform(X[train]), y[train])
        Models[i] = cv_predictor
        y_idx = np.append(y_idx, idx[test])
        y_pred = np.r_[y_pred, cv_predictor.predict(selector.transform(X[test]))]
        y_true = np.append(y_true, y[test])
    return y_idx, y_true, y_pred, Models


def Plot_True_vs_Pred(y_true, y_pred, title='XGB_Reg', low=None, up=None, figure_file=None):
    sns.set(style='whitegrid', palette='muted', color_codes=True)
    low = low if low!=None else -12
    up = up if up!=None else 12
    fig = plt.figure(figsize=(10,8))
    ax = plt.subplot(1, 1, 1)
    plt.scatter(y_true, y_pred,  color='green')
    plt.plot((low,up), (low,up), color='blue', linewidth=3)
    plt.tick_params(axis='both', which='major', labelsize=18)
    if title:
        plt.title(label=title, fontsize=28)
    plt.xlabel(xlabel=r'$\Delta\Delta G_{\rm DFT}$ (kcal/mol)', fontsize=24)
    plt.ylabel(ylabel=r'$\Delta\Delta G_{\rm ML}$ (kcal/mol)', fontsize=24)
    R2 = r2_score(y_true, y_pred)
    MAE = mean_absolute_error(y_true, y_pred)
    MSE = mean_squared_error(y_true, y_pred)
    plt.text((low+up)/2+3, low+2, s='R$^2$ = %.3f\nMAE = %.2f kcal/mol\nMSE = %.2f kcal/mol\n'%(R2, MAE, MSE),
            fontdict={'size': '20', 'color': 'black'})
    
    #plt.subplots_adjust(wspace =0.3, hspace =0.3)#调整子图间距
    if figure_file != None:
        plt.savefig(figure_file, dpi=300)
    plt.show()


def Barh_Feature_Ranking(selector, best_k=15, figure_file=None):
    sns.set(style="ticks")
    colors_lib = np.array(['salmon', 'orange', 'blue', 'purple', 'darkgreen','red', 'tan', 'cyan', 'violet', 'lightgreen'])

    def get_EQBVN(astr):
        EQBVN2Num_dict = {'E': 0, 'Q': 1, 'B': 2, 'V': 3, 'N': 4}
        EQBVN = re.findall(r'E|Q|B|V|N', astr)
        is_of_R = 1 if re.match(r'R}',astr[-3:]) else 0
        Num = EQBVN2Num_dict[EQBVN[0]] + is_of_R*len(EQBVN2Num_dict)
        return Num

    features = get_PhyChem_features_name()
    features_selected = [f for f, s in zip( features, selector.support_) if s]
    features_ranking = np.array(
        [features_selected, selector.estimator_.feature_importances_]).T
    features_ranking = pd.DataFrame(features_ranking, columns=[
        'Feature name', 'scores']).sort_values(by=['scores'], ascending=False)
    if best_k == None:
        best_k = features_ranking.shape[0]
    
    features_ranking['scores'] = features_ranking['scores'].astype(float)
    color_idx = [get_EQBVN(x) for x in features_ranking.iloc[:best_k, 0]]
    colors = colors_lib.take(color_idx)

    plt.rcdefaults()
    #fig, ax = plt.subplots(figsize=(10, 8))
    fig, ax = plt.subplots(figsize=(10, best_k//2+1))

    ax.barh(np.arange(best_k), features_ranking.iloc[:best_k, 1], align='center',
            color=colors, alpha=0.8)
    _R = 'R\N{MIDDLE DOT}'
    labels = ['FMO Energy (Arene)', 'Atomic Charge (Arene)','Bond Order (Arene)', 'Buried Volumn (Arene)', 'NICS (Arene)',
           'FMO Energy (%s)'%_R, 'Atomic Charge (%s)'%_R,'Bond Order (%s)'%_R, 'Buried Volumn (%s)'%_R, 'NICS (%s)'%_R,]
    color_unique, idx_c = np.unique(color_idx, return_index=True)
    for ci in color_unique[np.argsort(idx_c)]:
        if ci < len(colors_lib)/2:
            ax.barh(np.argwhere(np.array(color_idx) == ci).flatten(),
                    0, color=colors_lib[ci], alpha=0.8, label=labels[ci])
    for ci in color_unique[np.argsort(idx_c)]:
        if ci >= len(colors_lib)/2:
            ax.barh(np.argwhere(np.array(color_idx) == ci).flatten(),
                    0, color=colors_lib[ci], alpha=0.8, label=labels[ci])

    plt.tick_params(axis='both', which='major', labelsize=18)
    ax.set_yticks(np.arange(best_k))
    ax.set_yticklabels(features_ranking.iloc[:best_k, 0], fontsize=18)
    ax.invert_yaxis()  # labels read top-to-bottom
    #x_max, x_min = float(features_ranking['scores'].max()), float(features_ranking['scores'].min())
    #ax.set_xticks(np.linspace(x_min, x_min, 5))
    #ax.set_xticklabels(tuple(np.linspace(x_min, x_max, 5)))
    ax.set_xlabel('Feature Importances Scores', fontsize=24)
    #ax.set_title('Feature importances Ranking', fontsize=30)
    ax.legend(fontsize=18, loc='lower right')
    plt.subplots_adjust(left=0.20)

    if figure_file != None:
        plt.savefig(figure_file, dpi=300)
    plt.show()


def get_RFECV_result(model, X, y, n_jobs):
    min_f = 1
    
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    score_fun = skl.metrics.make_scorer(skl.metrics.r2_score)
    selector = RFECV(model, step=1, min_features_to_select=min_f,
                      cv=cv, scoring=score_fun, n_jobs=n_jobs)
    selector = selector.fit(X, y)
    return selector


def plot_learning_curve(train_sizes, train_scores, test_scores, fit_times, title, ylim=None, figure_file=None):
    sns.set(style='whitegrid', palette='muted', color_codes=True)
    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    
    axes.tick_params(axis='both', which='major', labelsize=18)
    if title:
        axes.set_title(title, fontsize=28)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples", fontsize=24)
    axes.set_ylabel("Score (R$^2$)", fontsize=24)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes.grid(b=True)
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes.legend(loc="lower right", fontsize=18)
    if figure_file != None:
        plt.savefig(figure_file, dpi=300)
    plt.show()

    # Plot n_samples vs fit_times
    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    axes.grid(b=True)
    axes.tick_params(axis='both', which='major', labelsize=18)
    axes.plot(train_sizes, fit_times_mean, 'o-')
    axes.fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes.set_xlabel("Training examples", fontsize=24)
    axes.set_ylabel("fit_times (s)", fontsize=24)
    axes.set_title("Scalability of the model", fontsize=28)
    plt.show()

    # Plot fit_time vs score
    fig, axes = plt.subplots(1, 1, figsize=(10, 8))
    axes.grid(b=True)
    axes.tick_params(axis='both', which='major', labelsize=18)
    axes.plot(fit_times_mean, test_scores_mean, 'o-')
    axes.fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes.set_xlabel("fit_times (s)", fontsize=24)
    axes.set_ylabel("Score ($R^2$)", fontsize=24)
    axes.set_title("Performance of the model", fontsize=28)
    #plt.subplots_adjust(wspace =0.3, hspace =0.3)#调整子图间距
    plt.show()

    
def is_HighSelectivity(DDG, neg_DDG_cutoff=None, scale_cutoff=None):
    def scale2DDG(x): return -math.log(x)*(298.15*8.31451)/(4.184*1000)
    def DDG2scale(x): return math.exp(-x*4.184*1000/(298.15*8.31451))
    if not neg_DDG_cutoff and scale_cutoff:
        neg_DDG_cutoff = -scale2DDG(scale_cutoff)
    return 1 if DDG >= neg_DDG_cutoff else 0


def get_Regioselectivity_sort(Group_data, neg_DDG_cutoff=1.42):
    Ars = Group_data.index.levels[0]
    Rs = Group_data.columns
    ArR_pool = []
    ArR_sel = []
    for R in Rs[:]:
        for Ar in Ars[:]:
            ArR_mol = Group_data.loc[Ar, R]
            loc1 = ArR_mol.index.levels[0].take(ArR_mol.index.codes[0])
            loc2 = ArR_mol.index.levels[1].take(ArR_mol.index.codes[1])
            loc_index = list(np.unique(np.append(loc1, loc2)))
            ArR_Matrix = pd.DataFrame(
                np.zeros((len(loc_index), len(loc_index))), index=loc_index, columns=loc_index)
            flag = False if (loc1.values - loc2.values).sum() == 0 else True
            for i in range(len(loc1)):
                value = ArR_mol.loc[loc1[i], loc2[i]]
                ArR_Matrix.loc[loc1[i], loc2[i]] = value
                if flag:
                    ArR_Matrix.loc[loc2[i], loc1[i]] = - value
            
            _nan_M = ArR_Matrix.isna().sum(axis=0)
            _nan_M_set = _nan_M.unique()
            if len(_nan_M_set) == 1 and _nan_M_set[0] == len(loc_index)-1:
                continue
            nan_idx_x = _nan_M.idxmax()
            nan_idx_y = ArR_Matrix.isna().sum(axis=1).idxmax()
            if _nan_M[nan_idx_x] == len(loc_index)-1 and nan_idx_x == nan_idx_y:
                ArR_Matrix.loc[nan_idx_x, nan_idx_y] = np.nan
                loc_index.remove(nan_idx_x)
            ArR_Matrix.dropna(axis=0, how='all',inplace=True)
            ArR_Matrix.dropna(axis=1, how='all',inplace=True)
            
            if len(loc_index) <= 1:
                continue
            ArR_ave = ArR_Matrix.mean(axis=1)
            ArR_ave -= ArR_ave.min()
            ArR_ave.sort_values(ascending=False, inplace=True)
            ArR_ave = ArR_ave.round(decimals=2)
            tmp = [[Ar, R, ArR_ave.index.to_numpy()[x], ArR_ave.to_numpy()[x]]
                   for x in range(len(loc_index))]
            ArR_pool += tmp
            degree = is_HighSelectivity(
                ArR_ave.iloc[0]-ArR_ave.iloc[1], neg_DDG_cutoff=neg_DDG_cutoff)
            tmp_sel = [[Ar, R, ArR_ave.index.to_numpy()[0], degree]]
            ArR_sel += tmp_sel
    column = ['Ar', 'R', 'loc', 'neg_DDG']
    ArR_df = pd.DataFrame(ArR_pool, columns=column)
    column_sel = ['Ar', 'R', 'sel@loc', 'degree']
    ArR_sel_df = pd.DataFrame(ArR_sel, columns=column_sel)
    return ArR_df, ArR_sel_df


def actual_vs_pred(df, neg_DDG_cutoff=1.42):
    Group_data_true = df['y_true'].groupby(
        by=[df['Ar'], df['loc1'], df['loc2'], df['R']]).mean().unstack()
    Group_data_pred = df['y_pred'].groupby(
        by=[df['Ar'], df['loc1'], df['loc2'], df['R']]).mean().unstack()

    ArR_df_true, ArR_sel_df_true = get_Regioselectivity_sort(
        Group_data_true, neg_DDG_cutoff)
    ArR_df_pred, ArR_sel_df_pred = get_Regioselectivity_sort(
        Group_data_pred, neg_DDG_cutoff)
    ArR_df = pd.merge(left=ArR_df_true, right=ArR_df_pred, on=[
                      'Ar', 'R', 'loc'], suffixes=('_true', '_pred'))
    ArR_df_sel = pd.merge(left=ArR_sel_df_true, right=ArR_sel_df_pred, on=[
        'Ar', 'R'], suffixes=('_true', '_pred'))
    return ArR_df, ArR_df_sel


def get_accurancy(df):
    def func(x): return 0 if x == 0 else 1
    tmp = df['sel@loc_true'] - df['sel@loc_pred']
    df['site_error'] = tmp.apply(func)
    tmp = df['degree_true'] - df['degree_pred']
    df['degree_error'] = tmp.apply(func)

    site_acc = df['site_error'].value_counts()[0]/df.shape[0]
    degree_acc = df[((df['site_error'] == 0) & (df['degree_error'] == 0)) | \
                ((df['degree_true'] == 0) & (df['degree_pred'] == 0))
                ]['degree_error'].value_counts()[0]/df.shape[0]
    site_acc = np.round(site_acc, decimals=4)
    degree_acc = np.round(degree_acc, decimals=4)
    return df, site_acc, degree_acc

def merge_csv(path):
    df = None
    for i, csv_f in enumerate(glob(path)):
        csv_fn = os.path.basename(csv_f)
        print(csv_fn)
        if i == 0:
            df = pd.read_csv(csv_f)
            print(df.shape)
        else:
            tmp_df = pd.read_csv(csv_f)
            print(tmp_df.shape)
            df = pd.concat([df, tmp_df], axis=0)
    assert type(df)==pd.DataFrame, "Can't find any target csv file in the given path "
    return df


def df_2heatmap(data):
    sns.set(style='darkgrid', palette='muted', color_codes=True)
    fig, ax = plt.subplots(figsize=(15, 8))
    
    ax = sns.heatmap(data=data, annot=True, fmt='.3f',
                     cmap="YlGnBu", ax=ax, vmin=0.80, vmax=1)
    # fix for mpl bug that cuts off top/bottom of seaborn viz
    #b, t = ax.get_ylim() # discover the values for bottom and top
    #b += 0.5 # Add 0.5 to the bottom
    #t -= 0.5 # Subtract 0.5 from the top
    #ax.set_ylim(b, t) # update the ylim(bottom, top) values
    
    #设置坐标字体方向
    #label_y = ax.get_yticklabels()
    #plt.setp(label_y, rotation=45, horizontalalignment='right')
    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=45, horizontalalignment='right')

    plt.show()
    
    