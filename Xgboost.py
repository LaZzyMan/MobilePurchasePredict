import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
import matplotlib.pylab as plt
import time
from pylab import mpl
import csv

mpl.rcParams['font.sans-serif'] = ['FangSong']
mpl.rcParams['axes.unicode_minus'] = False


def modelfit(alg, train_X, train_Y, test, useTrainCV=True, cv_folds=5, early_stopping_rounds=100):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(train_X, label=train_Y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
        alg.set_params(n_estimators=cvresult.shape[0])
    alg.fit(train_X, train_Y, eval_metric='auc')
    dtrain_predictions = alg.predict(train_X)
    dtrain_predprob = alg.predict_proba(train_X)[:, 1]
    test_predictions = alg.predict(test)
    test_predprob = alg.predict_proba(test)[:, 1]
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(train_Y, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(train_Y, dtrain_predprob))
    print(test_predictions)
    print(test_predprob)
    alg.get_booster().save_model('xgboost_3.model')
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()
    return {'train_pred': dtrain_predictions, 'train_pb': dtrain_predprob, 'test_pred': test_predictions, 'test_pb': test_predprob}


def optimize_n_estimators(data, test):
    train_X = data.drop(['next_step'], axis=1)
    train_Y = data.pop('next_step')
    xgb1 = XGBClassifier(
        learning_rate=0.01,
        n_estimators=1000,
        max_depth=6,
        min_child_weight=6,
        gamma=0.0,
        subsample=0.9,
        colsample_bytree=0.7,
        objective='binary:logistic',
        scale_pos_weight=13,
        seed=1000,
        max_delta_step=1
    )
    return modelfit(xgb1, train_X, train_Y, test=test)


def optimize_wieght_depth(data):
    train_X = data.drop(['next_step'], axis=1)
    train_Y = data.next_step
    param_test = {
        # 'max_depth': list(range(3, 10, 2)),
        'min_child_weight': list(range(1, 6, 2))
    }
    gsearch = GridSearchCV(estimator=XGBClassifier(learning_rate=0.01, n_estimators=233, max_depth=5,
                                                   min_child_weight=3, gamma=0, subsample=0.9, colsample_bytree=0.7,
                                                   objective='binary:logistic', scale_pos_weight=9,
                                                   seed=27, silent=False, max_delta_step=1, reg_alpha=225, reg_lambda=1),
                           param_grid=param_test, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch.fit(train_X, train_Y)
    print(gsearch.grid_scores_)
    print(gsearch.best_params_)
    print(gsearch.best_score_)


def optimize_gamma(data):
    train_X = data.drop(['next_step'], axis=1)
    train_Y = data.next_step
    param_test = {
        'gamma': [i / 10.0 for i in range(0, 5)]
    }
    gsearch = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=114, max_depth=3,
                                                   min_child_weight=3, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                   objective='binary:logistic', scale_pos_weight=50,
                                                   seed=27, silent=False),
                           param_grid=param_test, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch.fit(train_X, train_Y)
    print(gsearch.grid_scores_)
    print(gsearch.best_params_)
    print(gsearch.best_score_)


def optimize_sample(data):
    train_X = data.drop(['next_step'], axis=1)
    train_Y = data.next_step
    param_test = {
        'subsample': [i / 10.0 for i in range(6, 10)],
        'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    }
    gsearch = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=114, max_depth=3,
                                                   min_child_weight=3, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                   objective='binary:logistic', scale_pos_weight=50,
                                                   seed=27, silent=False),
                           param_grid=param_test, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch.fit(train_X, train_Y)
    print(gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_)


def optimize_alpha(data):
    train_X = data.drop(['next_step'], axis=1)
    train_Y = data.next_step
    param_test = {
        'reg_alpha': [200, 220, 225, 240, 250]
    }
    gsearch = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=114, max_depth=3,
                                                   min_child_weight=3, gamma=0, subsample=0.9, colsample_bytree=0.7,
                                                   objective='binary:logistic', scale_pos_weight=50,
                                                   seed=27, silent=False),
                           param_grid=param_test, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch.fit(train_X, train_Y)
    print(gsearch.grid_scores_)
    print(gsearch.best_params_)
    print(gsearch.best_score_)


def optimize_lambda(data):
    train_X = data.drop(['next_step'], axis=1)
    train_Y = data.next_step
    param_test = {
        'max_delta_step': [0, 1, 2, 3, 4, 5]
    }
    gsearch = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=114, max_depth=3,
                                                   min_child_weight=3, gamma=0, subsample=0.9, colsample_bytree=0.7,
                                                   objective='binary:logistic', scale_pos_weight=9,
                                                   seed=27, silent=False, reg_alpha=225, reg_lambda=1),
                           param_grid=param_test, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch.fit(train_X, train_Y)
    print(gsearch.grid_scores_)
    print(gsearch.best_params_)
    print(gsearch.best_score_)


def simple_train(data):
    train, verify = train_test_split(data, test_size=0.3, random_state=1)
    train_X = train.drop(['next_step'], axis=1)
    train_Y = train.next_step
    verify_X = verify.drop(['next_step'], axis=1)
    verify_Y = verify.next_step
    xg_train = xgb.DMatrix(train_X, label=train_Y, missing=-1)
    xg_verify = xgb.DMatrix(verify_X, label=verify_Y, missing=-1)
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'gamma': 0.1,
        'max_depth': 10,
        'lambda': 1,
        'alpha': 225,
        'subsample': 0.7,
        'colsample_bytree': 1,
        'min_child_weight': 6,
        'silent': 0,
        'eta': 0.01,
        'seed': 1000,
        'eval_metric': 'auc',
        'scale_pos_weight': 10,
        'max_delta_step': 1
    }
    watchlist = [(xg_train, 'train'), (xg_verify, 'verify')]
    num_round = 250
    bst = xgb.train(params, xg_train, num_round, watchlist)
    pred = bst.predict(xg_verify)
    error_rate = np.sum(pred != verify_Y) / verify_Y.shape[0]
    print('Test error using softmax = {}'.format(error_rate))
    # xgb.plot_importance(bst)
    # plt.show()

    xgb.plot_tree(bst, num_trees=1)
    plt.show()
    cost_time = time.time() - start_time
    print("XGBoost success!", '\n', "Cost Time:", cost_time, "(s)")


def model_train(data):
    with open('fmap.csv', encoding='utf-8') as f:
        fmap = f.read().strip().split(',')
        with open('fmap.txt', 'w', encoding='utf-8') as w:
            count = 0
            for feature in fmap:
                w.write(str(count) + '\t' + feature + '\tq\n')
                count += 1
            w.close()
        bst = xgb.Booster()
        bst.load_model('xgboost_3.model')
        score_weight = bst.get_score('fmap.txt', 'weight')
        score_gain = bst.get_score('fmap.txt', 'gain')
        score_importance = bst.get_fscore('fmap.txt')
        with open('./result/feature_result_3.csv', 'w', encoding='utf-8', newline='') as w:
            writer = csv.writer(w)
            writer.writerow(['f_name', 'score_weight', 'score_gain', 'score_importance'])
            for f_name, importance in score_importance.items():
                split_value = bst.get_split_value_histogram(f_name, 'fmap.txt')
                writer.writerow([f_name, score_weight[f_name], score_gain[f_name], score_importance[f_name]])
            w.close()

        # fig, ax = plt.subplots()
        # fig.set_size_inches(60, 30)
        # xgb.plot_tree(bst, ax=ax, num_trees=0, rankdir='LR')

        # fig.savefig('tree_2.jpg', dpi=100)
        # fig.show()
        xgb.to_graphviz(bst, num_trees=1, rankdir='RL').render()
        # feat_imp = pd.Series(fscore).sort_values(ascending=False)
        # feat_imp.plot(kind='bar', title='Feature Importances')
        # plt.ylabel('Feature Importance Score')
        # plt.show()
        # f.close()


def predict(train, test):
    train_X = train.drop(['next_step'], axis=1)
    train_Y = train.pop('next_step')
    bst = XGBClassifier()
    bst.fit(train_X, train_Y, eval_metric='auc')
    bst.get_booster().load_model('xgboost.model')
    pred = bst.predict(test)
    pred_prob = bst.predict_proba(test)
    print(pred)
    print(pred_prob)


if __name__ == '__main__':
    start_time = time.time()
    data = pd.read_csv('user_2_train.csv')
    train_id = data.id
    train_real = data.next_step
    data = data.drop(['id'], axis=1)
    test_data = pd.read_csv('user_2_test.csv')
    test_id = test_data.id
    test = test_data.drop(['id', 'next_step'], axis=1)

    # result = optimize_n_estimators(data, test)
    # optimize_wieght_depth(data)
    # optimize_gamma(data)
    # optimize_lambda(data)
    # optimize_alpha(data)
    # optimize_sample(data)
    # simple_train(data)
    model_train(data)
    # predict(data, test)
    # with open('./result/test_result_1.csv', 'w', encoding='utf-8', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['id', 'prediction', 'predict_prob'])
    #     for id, pred, pb in zip(list(test_id), result['test_pred'], result['test_pb']):
    #         writer.writerow([id, pred, pb])
    #     f.close()
    # with open('./result/train_result_1.csv', 'w', encoding='utf-8', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['id', 'prediction', 'real', 'predict_prob'])
    #     for id, pred, real, pb in zip(list(train_id), list(train_real), result['train_pred'], result['train_pb']):
    #         writer.writerow([id, pred, real, pb])
    #     f.close()
