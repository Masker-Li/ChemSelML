from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR, LinearSVR
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from xgboost import XGBRegressor

#XGBoost
XGB_R = XGBRegressor(
    max_depth=5, 
    learning_rate=0.1, 
    n_estimators=100, 
    verbosity=1, 
    objective='reg:squarederror', 
    booster='gbtree', 
    tree_method='auto', 
    n_jobs=2, 
    gamma=0, 
    min_child_weight=1, 
    max_delta_step=0, 
    subsample=1, 
    colsample_bytree=1, 
    colsample_bylevel=1, 
    colsample_bynode=1, 
    reg_alpha=0, 
    reg_lambda=1, 
    scale_pos_weight=1, 
    base_score=0.5, 
    random_state=0, 
    missing=None, 
    num_parallel_tree=1, 
    importance_type='gain')

# RandomForest
RF_R = RandomForestRegressor(
    n_estimators=100,
    criterion='mse',
    #max_depth=50,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features='auto',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    bootstrap=True,
    oob_score=False,
    n_jobs=2,
    random_state=0,
    verbose=0,
    warm_start=False)

# AdaBoost
Ada_R = AdaBoostRegressor(
    base_estimator=None, 
    n_estimators=50, 
    learning_rate=1.0, 
    loss='linear',  # ‘linear’, ‘square’, ‘exponential’
    random_state=None)

# Gradient Boosting
GB_R = GradientBoostingRegressor(
    loss='ls', 
    learning_rate=0.1, 
    n_estimators=100, 
    subsample=1.0, 
    criterion='friedman_mse', 
    min_samples_split=2, 
    min_samples_leaf=1, 
    min_weight_fraction_leaf=0.0, 
    max_depth=3, 
    min_impurity_decrease=0.0, 
    min_impurity_split=None, 
    init=None, 
    random_state=None, 
    max_features=None, 
    alpha=0.9, 
    verbose=0, 
    max_leaf_nodes=None, 
    warm_start=False, 
    presort='auto', 
    validation_fraction=0.1, 
    n_iter_no_change=None, 
    tol=0.0001)

NN_R = MLPRegressor(
    hidden_layer_sizes=(100, 100,), 
    activation='relu', 
    solver='adam', 
    alpha=0.0001, 
    batch_size='auto', 
    learning_rate='constant', 
    learning_rate_init=0.001, 
    power_t=0.5, 
    max_iter=200, 
    shuffle=True, 
    random_state=None, 
    tol=0.0001, 
    verbose=False, 
    warm_start=False, 
    momentum=0.9, 
    nesterovs_momentum=True, 
    early_stopping=False, 
    validation_fraction=0.1, 
    beta_1=0.9, 
    beta_2=0.999, 
    epsilon=1e-08, 
    n_iter_no_change=10, 
    max_fun=15000)

# Tree
DTree_R = DecisionTreeRegressor(
    criterion='mse', 
    splitter='best', 
    max_depth=None, 
    min_samples_split=2, 
    min_samples_leaf=1, 
    min_weight_fraction_leaf=0.0, 
    max_features=None, 
    random_state=None, 
    max_leaf_nodes=None, 
    min_impurity_decrease=0.0, 
    min_impurity_split=None, 
    presort=False)

# LR
LR_R = linear_model.LinearRegression(
    fit_intercept=True, 
    normalize=False, 
    copy_X=True, 
    n_jobs=2)

# Lasso
Lasso_R = linear_model.Lasso(
    alpha=1.0, 
    fit_intercept=True, 
    normalize=False, 
    precompute=False, 
    copy_X=True, 
    max_iter=1000, 
    tol=0.0001, 
    warm_start=False, 
    positive=False, 
    random_state=None, 
    selection='cyclic')

# Ridge Regression
Ridge_R = linear_model.Ridge(
    alpha=.5, 
    fit_intercept=True, 
    normalize=False, 
    copy_X=True, 
    max_iter=None, 
    tol=0.001, 
    solver='auto', 
    random_state=None)

# Bayesian Ridge Regression
BRR_R = linear_model.BayesianRidge(
    n_iter=300, 
    tol=0.001, 
    alpha_1=1e-06, 
    alpha_2=1e-06, 
    lambda_1=1e-06, 
    lambda_2=1e-06, 
    compute_score=False, 
    fit_intercept=True, 
    normalize=False, 
    copy_X=True, 
    verbose=False)

# KRR
KRR_R = KernelRidge(
    alpha=1, 
    kernel='linear', 
    gamma='auto', 
    degree=3, 
    coef0=1, 
    kernel_params=None)

# SVR
SVR_R = SVR(
    kernel='rbf', # {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}
    degree=3, 
    gamma='auto', 
    coef0=0.0, 
    tol=0.001, 
    C=1.0, 
    epsilon=0.1, 
    shrinking=True, 
    cache_size=200, 
    verbose=False, 
    max_iter=-1)

# linear SVR
LSVR_R = LinearSVR(
    epsilon=0.0, 
    tol=0.0001, 
    C=1.0, 
    loss='epsilon_insensitive', 
    fit_intercept=True, 
    intercept_scaling=1.0, 
    dual=True, 
    verbose=0, 
    random_state=None, 
    max_iter=5000)

# SGD
SGD_R = linear_model.SGDRegressor(
    loss='squared_loss', 
    penalty='l2', 
    alpha=0.0001, 
    l1_ratio=0.15, 
    fit_intercept=True, 
    max_iter=1000, 
    tol=0.001, 
    shuffle=True, 
    verbose=0, 
    epsilon=0.1, 
    random_state=None, 
    learning_rate='invscaling', 
    eta0=0.01, 
    power_t=0.25, 
    early_stopping=False, 
    validation_fraction=0.1, 
    n_iter_no_change=5, 
    warm_start=False, 
    average=False)

#KNR
KNR_R = KNeighborsRegressor(
    n_neighbors=5, 
    weights='uniform', 
    algorithm='auto', 
    leaf_size=30, 
    p=2, 
    metric='minkowski', 
    metric_params=None, 
    n_jobs=2)

# GPR
kernel = DotProduct() + WhiteKernel()
GPR_R = GaussianProcessRegressor(
    kernel=None, 
    alpha=1e-10, 
    optimizer='fmin_l_bfgs_b', 
    n_restarts_optimizer=0, 
    normalize_y=False, 
    copy_X_train=True, 
    random_state=None)

def get_regressor_lib(keys=None):
    '''keys: None, keys or list'''
    assert keys == None or type(keys) in [str,list
              ], 'keys should be None, a string or a list!'
    
    reg_lib = { 'XGB_R' : XGB_R,  'RF_R': RF_R,   'Ada_R': Ada_R, 'GB_R' : GB_R, 'DTree_R': DTree_R, 'LR_R' : LR_R, 
            'Lasso_R' : Lasso_R, 'Ridge_R': Ridge_R, 'BRR_R': BRR_R, 'KRR_R'  : KRR_R,   'SVR_R': SVR_R, 
            'LSVR_R'  : LSVR_R,  'SGD_R'  : SGD_R,   'KNR_R': KNR_R, 'GPR_R'  : GPR_R,    'NN_R': NN_R }
    
    
    if keys == None:
        return reg_lib
    elif type(keys) == str:
        assert keys in reg_lib.keys()
        return reg_lib[keys]
    elif type(keys) == list:
        for key in keys:
            assert key in reg_lib.keys()
        return [reg_lib[key] for key in keys]

