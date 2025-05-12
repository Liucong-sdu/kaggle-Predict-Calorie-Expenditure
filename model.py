import warnings
warnings.filterwarnings('ignore')
from lightgbm import LGBMRegressor as LGBMR, early_stopping
from xgboost import XGBRegressor as XGBR
from catboost import CatBoostRegressor as CBR
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.metrics import mean_absolute_error

class cfg:
    trainfilepath = "train_processed.csv"
    testfilepath = "test_processed.csv"
    outfilepath = "myoutput"
    state = 8
    n_iter = 20  # 每个模型的参数搜索次数
    early_stop_rounds = 50

class future_engineer:
    traindata = pd.read_csv(cfg.trainfilepath)
    testdata = pd.read_csv(cfg.testfilepath)
    headerstrian = set(traindata.columns)
    headerstest = set(testdata.columns)
    target = headerstrian.symmetric_difference(headerstest).pop()
    print("获取的目标列是：", target)

class dataset_split:
    # 划分训练集和验证集（保持原始测试集不变）
    X_full = future_engineer.traindata.drop([future_engineer.target], axis=1)
    Y_full = future_engineer.traindata[future_engineer.target]
    X_train, X_val, Y_train, Y_val = train_test_split(X_full, Y_full, test_size=0.1, random_state=cfg.state)
    X_test = future_engineer.testdata
    print("数据分隔完成（含验证集）")

class HyperparameterSearch:
    @staticmethod
    def lgbm_search():
        param_dist = {
            'learning_rate': [0.005, 0.01, 0.02],
            'max_depth': [4,5,6,7,8],
            'colsample_bytree': [0.8, 0.9, 0.95],
            'reg_alpha': [0.001, 0.01, 0.1],
            'reg_lambda': [0.001, 0.01, 0.1]
        }
        best_score = np.inf
        best_model = None
        
        for params in ParameterSampler(param_dist, n_iter=cfg.n_iter, random_state=cfg.state):
            model = LGBMR(
                objective="regression",
                n_estimators=1000,
                **params,
                random_state=cfg.state,
                verbosity=-1
            )
            model.fit(
                dataset_split.X_train, dataset_split.Y_train,
                eval_set=[(dataset_split.X_val, dataset_split.Y_val)],
                callbacks=[early_stopping(cfg.early_stop_rounds)],
                
            )
            pred = model.predict(dataset_split.X_val)
            score = mean_absolute_error(dataset_split.Y_val, pred)
            if score < best_score:
                best_score = score
                best_model = model
        print(f"LGBM 最佳验证MAE: {best_score:.4f}")
        print(best_model)
        return best_model

    @staticmethod
    def xgb_search():
        param_dist = {
            'learning_rate': [0.005, 0.01, 0.02],
            'max_depth': [4, 5,6, 7,8],
            'colsample_bytree': [0.8, 0.9, 0.95],
            'reg_alpha': [0.001, 0.01, 0.1],
            'gamma': [0, 0.1, 0.2]
        }
        best_score = np.inf
        best_model = None

        for params in ParameterSampler(param_dist, n_iter=cfg.n_iter, random_state=cfg.state):
            model = XGBR(
                objective="reg:squarederror",
                n_estimators=1000,
                
                **params,
                random_state=cfg.state,
                n_jobs=4,
                tree_method="gpu_hist",
                predictor="gpu_predictor",
                gpu_id=0,
                eval_metric="mae",
            )
            model.fit(
                dataset_split.X_train, dataset_split.Y_train,
                eval_set=[(dataset_split.X_val, dataset_split.Y_val)],
                
                verbose=False,
                
                
            )
            pred = model.predict(dataset_split.X_val)
            score = mean_absolute_error(dataset_split.Y_val, pred)
            if score < best_score:
                best_score = score
                best_model = model
        print(f"XGB 最佳验证MAE: {best_score:.4f}")
        print(best_model)
        return best_model

    @staticmethod
    def cat_search():
        param_dist = {
                'learning_rate': [0.005, 0.01, 0.02],
                'depth': [4,5, 6,7, 8],  # 替换max_depth → depth[3,5,7](@ref)
                'rsm': [0.8, 0.9, 0.95],  # 替换colsample_bytree → max_features[3,5](@ref)
                'l2_leaf_reg': [0.1, 0.5, 1.0, 10]  # 扩展范围[3](@ref)
}
        best_score = np.inf
        best_model = None

        for params in ParameterSampler(param_dist, n_iter=cfg.n_iter, random_state=cfg.state):
            model = CBR(
                loss_function="RMSE",
                iterations=1000,
                **params,
                random_state=cfg.state,
                verbose=0,
                early_stopping_rounds=cfg.early_stop_rounds
            )
            model.fit(
                dataset_split.X_train, dataset_split.Y_train,
                eval_set=(dataset_split.X_val, dataset_split.Y_val),
                verbose=False
            )
            pred = model.predict(dataset_split.X_val)
            score = mean_absolute_error(dataset_split.Y_val, pred)
            if score < best_score:
                best_score = score
                best_model = model
        print(f"CatBoost 最佳验证MAE: {best_score:.4f}")
        print(best_model)
        return best_model

class model:
    models = {
        "LGBM": HyperparameterSearch.lgbm_search(),
        "XGB": HyperparameterSearch.xgb_search(),
        "CatBoost": HyperparameterSearch.cat_search()
    }
    print("\n所有模型训练完成")

class predict:
    predictions = {}
    for name, model in model.models.items():
        predictions[name] = model.predict(dataset_split.X_test)
        print(f"{name} 预测完成")

class writefile:
    test_ids = future_engineer.testdata["id"]
    for name, pred in predict.predictions.items():
        filename = os.path.join(cfg.outfilepath, f"{name}_submission.csv")
        pd.DataFrame({
            "id": test_ids,
            future_engineer.target: pred
        }).to_csv(filename, index=False)
        print(f"结果已保存至：{filename}")