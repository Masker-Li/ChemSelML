from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier


XGB_C = XGBClassifier(
    max_depth=5, 
    learning_rate=0.1, 
    n_estimators=200, 
    verbosity=1, 
    objective='multi:softprob', 
    booster='gbtree', 
    tree_method='auto', 
    n_jobs=2, 
    gpu_id=0, 
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
    missing=None)

Ridge_C = linear_model.RidgeClassifier(
    alpha=1.0, 
    fit_intercept=True, 
    normalize=False, 
    copy_X=True, 
    max_iter=None, 
    tol=0.001, 
    class_weight=None, 
    solver='auto', 
    random_state=None)

LogisticR_C = linear_model.LogisticRegression(
    penalty='l2', 
    dual=False, 
    tol=0.0001, 
    C=1.0, 
    fit_intercept=True, 
    intercept_scaling=1, 
    class_weight=None, 
    random_state=None, 
    solver='lbfgs', 
    max_iter=100, 
    multi_class='auto', 
    verbose=0, 
    warm_start=False, 
    n_jobs=2, 
    l1_ratio=None)

NN_C = MLPClassifier(
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

SVM_C = SVC(
    C=1.0, 
    kernel='rbf', 
    degree=3, 
    gamma='scale', 
    coef0=0.0, 
    shrinking=True, 
    probability=False, 
    tol=0.001, 
    cache_size=200, 
    class_weight=None, 
    verbose=False, 
    max_iter=-1, 
    decision_function_shape='ovr', 
    break_ties=False, 
    random_state=None)

NuSVC_C = NuSVC(
    nu=0.5, 
    kernel='rbf', 
    degree=3, 
    gamma='scale', 
    coef0=0.0, 
    shrinking=True, 
    probability=False, 
    tol=0.001, 
    cache_size=200, 
    class_weight=None, 
    verbose=False, 
    max_iter=-1, 
    decision_function_shape='ovr', 
    break_ties=False, 
    random_state=None)

LSVC_C = LinearSVC(
    penalty='l2', 
    loss='squared_hinge', 
    dual=True, 
    tol=0.0001,
    C=1.0, 
    multi_class='ovr', 
    fit_intercept=True, 
    intercept_scaling=1, 
    class_weight=None, 
    verbose=0, 
    random_state=None, 
    max_iter=1000)

SGD_C = SGDClassifier(
    loss='hinge', 
    penalty='l2', 
    alpha=0.0001, 
    l1_ratio=0.15, 
    fit_intercept=True, 
    max_iter=1000, 
    tol=0.001, 
    shuffle=True, 
    verbose=0, 
    epsilon=0.1, 
    n_jobs=2, 
    random_state=None, 
    learning_rate='optimal', 
    eta0=0.0, 
    power_t=0.5, 
    early_stopping=False, 
    validation_fraction=0.1, 
    n_iter_no_change=5, 
    class_weight=None, 
    warm_start=False, 
    average=False)

KN_C = KNeighborsClassifier(
    n_neighbors=5, 
    weights='uniform', 
    algorithm='auto', 
    leaf_size=30, 
    p=2, 
    metric='minkowski', 
    metric_params=None, 
    n_jobs=2)

RN_C = RadiusNeighborsClassifier(
    radius=1.0, 
    weights='uniform', 
    algorithm='auto', 
    leaf_size=30, 
    p=2, 
    metric='minkowski', 
    outlier_label=None, 
    metric_params=None, 
    n_jobs=2)

GP_C = GaussianProcessClassifier(
    kernel=None, 
    optimizer='fmin_l_bfgs_b', 
    n_restarts_optimizer=0, 
    max_iter_predict=100, 
    warm_start=False, 
    copy_X_train=True, 
    random_state=None, 
    multi_class='one_vs_rest', 
    n_jobs=2)

GNB_C = GaussianNB(
    priors=None, 
    var_smoothing=1e-09)

MNB_C = MultinomialNB(
    alpha=1.0, 
    fit_prior=True, 
    class_prior=None)

CNB_C = ComplementNB(
    alpha=1.0, 
    fit_prior=True, 
    class_prior=None, 
    norm=False)

BNB_C = BernoulliNB(
    alpha=1.0, 
    binarize=0.0, 
    fit_prior=True, 
    class_prior=None)

DT_C = DecisionTreeClassifier(
    criterion='gini', 
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
    class_weight=None, 
    presort='deprecated', 
    ccp_alpha=0.0)

RF_C = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=None,
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
    random_state=None,
    verbose=0,
    warm_start=False,
    class_weight=None,
    ccp_alpha=0.0,
    max_samples=None)

Ada_C = AdaBoostClassifier(
    base_estimator=None,
    n_estimators=50,
    learning_rate=1.0,
    algorithm='SAMME.R',
    random_state=None)

Bag_C = BaggingClassifier(
    base_estimator=None, 
    n_estimators=10, 
    max_samples=1.0, 
    max_features=1.0, 
    bootstrap=True, 
    bootstrap_features=False, 
    oob_score=False, 
    warm_start=False, 
    n_jobs=2, 
    random_state=None, 
    verbose=0)

GradB_C = GradientBoostingClassifier(
    loss='deviance', 
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
    verbose=0, 
    max_leaf_nodes=None, 
    warm_start=False, 
    presort='deprecated', 
    validation_fraction=0.1, 
    n_iter_no_change=None, 
    tol=0.0001, 
    ccp_alpha=0.0)

HGradB_C = HistGradientBoostingClassifier(
    loss='auto', 
    learning_rate=0.1, 
    max_iter=100, 
    max_leaf_nodes=31, 
    max_depth=None, 
    min_samples_leaf=20, 
    l2_regularization=0.0, 
    max_bins=255, 
    warm_start=False, 
    scoring=None, 
    validation_fraction=0.1, 
    n_iter_no_change=None, 
    tol=1e-07, 
    verbose=0, 
    random_state=None)

ExT_C = ExtraTreesClassifier(
    n_estimators=100, 
    criterion='gini', 
    max_depth=None, 
    min_samples_split=2, 
    min_samples_leaf=1, 
    min_weight_fraction_leaf=0.0, 
    max_features='auto', 
    max_leaf_nodes=None, 
    min_impurity_decrease=0.0, 
    min_impurity_split=None, 
    bootstrap=False, 
    oob_score=False, 
    n_jobs=2, 
    random_state=None, 
    verbose=0, 
    warm_start=False, 
    class_weight=None, 
    ccp_alpha=0.0, 
    max_samples=None)


def get_classifier_lib(keys=None):
    '''keys: None, keys or list'''
    assert keys == None or type(keys) in [str,list
              ], 'keys should be None, a string or a list!'
    
    clf_lib = {'Ridge_C': Ridge_C, 'LogR_C' : LogisticR_C, 'NN_C' : NN_C,     'SVM_C' : SVM_C,  'NuSVC_C' : NuSVC_C,  
           'LSVC_C' : LSVC_C,   'SGD_C' : SGD_C,       'KN_C' : KN_C,      'RN_C' : RN_C,      'GP_C' : GP_C, 
            'GNB_C' : GNB_C,    'MNB_C' : MNB_C,      'CNB_C' : CNB_C,    'BNB_C' : BNB_C,     'DT_C' : DT_C, 
             'RF_C' : RF_C,     'Ada_C' : Ada_C,      'Bag_C' : Bag_C,  'GB_C' : GradB_C,    'XGB_C': XGB_C,
         'HGradB_C' : HGradB_C, 'ExT_C' : ExT_C }
    
    if keys == None:
        return clf_lib
    elif type(keys) == str:
        assert keys in clf_lib.keys()
        return clf_lib[keys]
    elif type(keys) == list:
        for key in keys:
            assert key in clf_lib.keys()
        return [clf_lib[key] for key in keys]


