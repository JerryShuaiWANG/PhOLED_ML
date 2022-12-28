#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author       : Shengde Zhang, Shuai Wang
'''

import os
import hyperopt
import pandas as pd
import numpy as np
from numpy.random import RandomState
import sys
from QsarUtils import *
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, LogisticRegression
from hyperopt import hp
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor


def MainRegression(in_file_path, saved_dir, feature_selector_list, select_des_num_list,model_list, 
         search_max_evals, k=10, kfold_type="normal", random_state=0, search_metric="val_RMSE", greater_is_better=False,):
    """
    :param saved_dir:
    :param feature_selector_list: list/tup,'RFE', 'f_regression', or 'mutual_info_regression'
    :param select_des_num_list: int/tup,
    :param kfold_type: str, 'normal'(KFold)  / 'none'(all train)
    :param random_state:
    :return:
    """
    label = 'Exp. Kr'
    dataset = pd.read_csv("train.csv")
    dataset_test= pd.read_csv("test.csv")
    # dataset_test= pd.read_csv("external_test.csv")
    
 
    x = dataset.drop(columns=[label])
    train_X = x.to_numpy()
    train_y = dataset[label].to_numpy()
    
    test_x = dataset_test.drop(columns=[label])
    test_X = test_x.to_numpy()
    test_y = dataset_test[label].to_numpy()


    model = RegressionModel(random_state=random_state)
    model.LoadData(train_X, train_y, test_X, test_y)
    model.ScaleFeature(saved_dir=saved_dir,saved_file_note="_".join(''))
    model.KFoldSplit(k=k, kfold_type=kfold_type)

    def Search(params):
        nonlocal model
        nonlocal estimator
        nonlocal search_metric
        nonlocal greater_is_better
        print("#"*20)
        print("params: ",params)
        print("#"*20)
        feature_selector = "f_regression"
        select_des_num = 50
        if "feature_selector" in params:
            feature_selector = params["feature_selector"]
            del params["feature_selector"]
        if "select_des_num" in params:
            select_des_num = int(params["select_des_num"])
            del params["select_des_num"]
        if (model.feature_selector_name != feature_selector) or (model.feature_select_num != select_des_num):
            model.SelectFeature(feature_selector=feature_selector, select_des_num=select_des_num)
        else:
            pass
        model.Train(estimator,params=params)
        val_metric = model.all_metrics_df.loc["mean",search_metric]
        if greater_is_better:
            return -val_metric
        else:
            return val_metric

    lr_model = LinearRegression()
    lr_params = {}

    xgbr_model = XGBRegressor(objective='reg:squarederror', random_state=random_state)
    xgbr_params = {
        'gamma': hyperopt.hp.uniform("gamma", 0, 0.5),
        'max_depth': hyperopt.hp.uniformint('max_depth', 2, 15),
        'min_child_weight': hyperopt.hp.uniformint('min_child_weight', 1, 30),
        'colsample_bytree': hyperopt.hp.uniform('colsample_bytree', 0.5, 1),
        'subsample': hyperopt.hp.uniform('subsample', 0.5, 1),
        'learning_rate': hyperopt.hp.uniform('learning_rate', 0.001, 0.2),
        'n_estimators': hyperopt.hp.uniformint('n_estimators', 5, 500),
         'max_delta_step':hyperopt.hp.uniform('max_delta_step', 0.5, 1),
         'reg_alpha': hyperopt.hp.uniform('reg_alpha', 0, 0.5),
         'reg_lambda': hyperopt.hp.uniform('reg_lambda', 0.5, 1),
         'scale_pos_weight': hyperopt.hp.uniform('scale_pos_weight', 0, 0.2),
    }

    rfr_model = RandomForestRegressor(random_state=random_state)
    rfr_parms = {'n_estimators': hyperopt.hp.uniformint('n_estimators', 10, 500),
                'max_leaf_nodes': hyperopt.hp.uniformint('max_leaf_nodes', 10, 100),
                'min_samples_split': hyperopt.hp.uniformint('min_samples_split', 2, 10),
                'min_samples_leaf': hyperopt.hp.uniformint('min_samples_leaf', 1, 10),
                }

    svr_model = SVR()
    svr_params = {'C': hyperopt.hp.uniform("C", 1e-5, 1e2),
                  'gamma': hyperopt.hp.uniform("gamma", 1e-5, 1e2),
                  'epsilon': hyperopt.hp.uniform("epsilon", 1e-5, 1),
                  }


    knnD_designed_model = KNeighborsRegressor(algorithm="brute",weights=weights_function_KNND)#KNUni, KNDist
    knnD_designed_params  = {
        'n_neighbors': hyperopt.hp.uniformint('n_neighbors', 5, 20),
                'leaf_size': hyperopt.hp.uniformint('leaf_size', 1, 20),
                }
    knnD_model = KNeighborsRegressor(algorithm="brute",weights="distance")#KNUni, KNDist
    knnD_params  = {
        'n_neighbors': hyperopt.hp.uniformint('n_neighbors', 5, 20),
                'leaf_size': hyperopt.hp.uniformint('leaf_size', 1, 20),
                }
    knnU_model = KNeighborsRegressor(algorithm="auto", weights='uniform')#KNUni, KNDist
    knnU_params  = {'n_neighbors': hyperopt.hp.uniformint('n_neighbors', 5, 20),
                'leaf_size': hyperopt.hp.uniformint('leaf_size', 1, 20),
                }
    lgbm_model = LGBMRegressor()
    lgbm_params  = {'num_leaves': hyperopt.hp.uniformint('num_leaves', 2, 13),
            'learning_rate': hyperopt.hp.uniform('learning_rate', 0.00001, 0.2),
            'min_child_samples': hyperopt.hp.uniformint('min_child_samples', 0, 50),
            'max_depth': hyperopt.hp.uniformint('max_depth', 0, 13),
            'n_estimators': hyperopt.hp.uniformint('n_estimators', 10, 500),
            "bagging_fraction" :hyperopt.hp.uniform('bagging_fraction', 0.5, 1),
            }
    krr_model = KernelRidge()
    krr_params  = {'alpha': hyperopt.hp.uniform('alpha', 0, 3),
            }

    # pls_model = PLSRegression()
    ada_model = AdaBoostRegressor() 
    ada_params  = {'learning_rate': hyperopt.hp.uniform('learning_rate', 0.00001, 0.2),
        'n_estimators': hyperopt.hp.uniformint('n_estimators', 10, 500),
        }

    model_param_dict = {"LR": {"estimator": lr_model, "params": lr_params},
                        "XGB": {"estimator": xgbr_model, "params": xgbr_params},
                        "RF":{"estimator": rfr_model, "params": rfr_parms},
                        "SVM":{"estimator": svr_model, "params": svr_params},
                        "KNNU":{"estimator": knnU_model, "params": knnU_params},
                        "KNNDde":{"estimator": knnD_designed_model, "params": knnD_designed_params},
                        "KNND":{"estimator": knnD_model, "params": knnD_params},
                        "LGBM":{"estimator": lgbm_model, "params": lgbm_params},
                        "KRR":{"estimator": krr_model, "params": krr_params},
                        "ADA":{"estimator": ada_model, "params": ada_params},

                       }
    for m in model_list:
        estimator = model_param_dict[m]["estimator"]
        model_name = str(estimator)
        model_name = model_name[:model_name.find("(")]
        params_space = {"feature_selector": hyperopt.hp.choice('feature_selector',feature_selector_list),
                        "select_des_num":hyperopt.hp.choice("select_des_num",select_des_num_list)}
        params_space.update(model_param_dict[m]["params"])
        best_params = hyperopt.fmin(Search, space=params_space, algo=hyperopt.tpe.suggest,
                                    max_evals=search_max_evals,rstate=np.random.default_rng(random_state))#RandomState(ran)
        for key,value in params_space.items():
            if value.name == "int":
                best_params[key] = int(best_params[key])
        print("Best params: ",best_params)
        select_des_num = select_des_num_list[best_params["select_des_num"]]
        feature_selector = feature_selector_list[best_params["feature_selector"]]
        model.SelectFeature(feature_selector=feature_selector, select_des_num=select_des_num, saved_dir=saved_dir, saved_file_note=model_name)
        del best_params["select_des_num"]
        del best_params["feature_selector"]
        model.Train(estimator,params=best_params,saved_dir=saved_dir)
        model.all_metrics_df["model_name"] = model_name
        model.all_metrics_df["feature_num"] = select_des_num
        model.all_metrics_df["random_state"] = random_state
        metrics_out_file = os.path.join(saved_dir,"model_metrics.csv")
        model.all_metrics_df.to_csv(metrics_out_file,mode="a")
        model.SaveTotalModel(saved_dir=saved_dir,saved_file_note=random_state)
        model.GenerateBallTree(p=1,saved_dir=saved_dir)
def weights_function_KNND(x):
    return 1/(x+0.032)
if __name__ == "__main__":

###########Regression################
    t0 = time.time()
    data_dir = "./"
    in_file_name = ""

    in_file_path = os.path.join(data_dir, in_file_name)
    os.makedirs("./_/")

    random_state = 65
    feature_selector_list = ("RFE",)
    select_des_num_list = (46,)
    model_list = ("XGB","SVM","LGBM","RF","KNNDde","KNNU","KRR","ADA", )
    kfold_type = "normal"
    search_max_evals = 10
    search_metric = "val_RMSE"
    k = 10

    saved_dir = os.path.join(data_dir, "{}_{}".format(in_file_name[-11:],"_".join('')))
    print(saved_dir)


    MainRegression(in_file_path, saved_dir, feature_selector_list, select_des_num_list,model_list=model_list, search_max_evals=search_max_evals, 
                    k=k, kfold_type=kfold_type, random_state=random_state, search_metric=search_metric, greater_is_better=False)

    print("Time cost: {}".format(Sec2Time(time.time()-t0)))
