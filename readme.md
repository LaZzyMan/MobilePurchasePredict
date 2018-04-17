## 决策树算法总结
### 决策树算法适用范围：
- 定性特征较多
- 特征间较少存在相互关联
- 数据含有较多的缺失值
- 需要得到明确的特征对预测结果的影响方式以及决策过程
- 多种特征权重难以衡量或优化
----
### CART
**构造特点**
- 递归优化生成树
- 遍历所有Feature的所有分割点，寻找信息增量最大的特征分割方法，具有局部贪心的特点
- 信息增量的衡量可以采用熵测度和基尼不纯度测度，熵测度的收敛速度更慢
- 剪枝策略可以采用最小信息增量法合并信息增量小于阈值的非叶节点或删除权重和小于的叶节点

**缺点**
- 训练过程无法处理缺失值，需要删除大量样本
- 没有对不平衡样本的应对策略

----
### GBDT
**特点**
- 提升决策树算法是将boosting算法（将多个弱分类器分类函数累加得到强分类器）应用于决策树的结果，核心思想是使用前向分布进行残差拟合，学习下一棵回归树，再将所有回归树相加得到强分类函数。
- 引入了并行计算：在对某一特征进行分割时，利用多线程方法分段遍历分割点，各线程分别得出最优分割点，再有MPI模块处理得到整体最优分割点。

**sklearn使用CART和GBDT**
```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
# sklearn.datasets是sklearn中自带的测试数据，数据内容是对鸢尾花品种进行分类
from sklearn import datasets
# 读取数据，x为特征值矩阵，y为预测结果
iris=datasets.load_iris()
x=iris.data[:,:2]
y=iris.target
# 通过参数初始化决策树模型
CARTModel=DecisionTreeClassifier(max_depth=5)
GBDTModel=GradientBoostingClassifier(n_estimators=100)
# 模型拟合
CARTModel.fit(x,y)
GBDTModel.fit(x,y)
# 利用模型进行预测
CARTModel_pre=model1.predict(x)
GBDTModel_pre=model2.predict(x)
# 计算预测结果与实际结果的偏差
CARTResult=CARTModel_pre==y
GBDTResult=GBDTModel_pre==y
print '决策树训练集正确率%.2f%%'%np.mean(CARTResult*100)
print 'GDBT训练集正确率%.2f%%'%np.mean(GBDTResult*100)
```
有关模型的保存、读取、决策树的图形化输出以及详细参数设置参考：[sklearn相关文档][1]

详细算法原理数学推导参考:[梯度提升树GBDT原理][0]

-----
### XGBoost
**算法原理**

- 在损失函数中引入了模型复杂度，以叶子节点个数和叶子节点分数平方和作为衡量标准，并通过正则化系数alpha和lambda进行调节，目的是更好地防止过拟合（模型过于复杂容易产生过拟合）。
- 在缺失值的处理上，算法采用自我学习的方式解决缺失值问题。在模型训练阶段，在存在特征值缺失的节点上，通过分别尝试将缺失数据分到左右两个节点，计算损失函数值，选择较优的方向并记录下来，之后在预测中遇到缺失值时按照模型中记录的方向进行分配。
- 与GBDT相比，XGBoost使用了泰勒展开到二次项，分类精度有所提升；支持特征抽样（列抽样），降低了过拟合；同时支持逻辑回归、逻辑分类、多类分类以及分类概率输出。
- 详细算法原理参考：[Introduction to Boosted Trees][2]


**xgboost结合sklearn的使用**

需要使用的package
```python
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
import matplotlib.pylab as plt
import time # 用于记录运行时间
import csv # 用于创建fmap
```

训练和测试数据的导入
```python
# 训练数据中next_step为预测结果列，测试数据没有此列，测试和训练数据都含有id列，但不参与训练，只作为标识
train_data = pd.read_csv('user_2_train.csv')
# train_id = train_data.id
# train_real = train_data.next_step
train_data = train_data.drop(['id'], axis=1)
test_data = pd.read_csv('user_2_test.csv')
# test_id = test_data.id
test_data = test_data.drop(['id', 'next_step'], axis=1)
```
没有测试数据时的数据处理
```python
# 使用train_test_split方法将数据分割为训练集和测试集
train, verify = train_test_split(data, test_size=0.3, random_state=1)
train_X = train.drop(['next_step'], axis=1)
train_Y = train.next_step
verify_X = verify.drop(['next_step'], axis=1)
verify_Y = verify.next_step
# 将数据转为Xgboost使用的DMatrix格式，第一个参数为特征矩阵，label为预测结果列，missing为缺失值
xg_train = xgb.DMatrix(train_X, label=train_Y, missing=-1)
xg_verify = xgb.DMatrix(verify_X, label=verify_Y, missing=-1)
```
xbgoost提供了两种模型调用方式：

（1）使用xgboost进行训练
```python
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
'''
params:训练参数
num_boost_round:boosting迭代次数（决策树个数）
evals:训练过程中输出的评估值
early_stopping_rounds:连续迭该值次数后损失函数扔未明显下降，提前停止迭代
xgb_model:模型文件名，可直接读取模型文件
'''
bst = xgb.train(params, xg_train, num_round, watchlist)
pred = bst.predict(xg_verify)
error_rate = np.sum(pred != verify_Y) / verify_Y.shape[0]
print('Test error using softmax = {}'.format(error_rate))
bst.Booster().save_model('xgboost.model')
```
（2）使用XGBClassifier进行训练(与CART的用法相似)
```python
xgb = XGBClassifier(
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
xgb.fit(train_X, train_Y, eval_metric='auc')
# 分类预测
dtrain_predictions = xgb.predict(train_X)
# 概率预测
dtrain_predprob = xgb.predict_proba(train_X)[:, 1]
print("Accuracy : %.4g" % metrics.accuracy_score(train_Y, dtrain_predictions))
print("AUC Score (Train): %f" % metrics.roc_auc_score(train_Y, dtrain_predprob))
xgb.get_booster().save_model('xgboost_3.model')
```
参数的调整也分为两部分：

(1)利用交叉检验确定最佳决策树数量
```python
xgb_param = xgb.get_xgb_params()
xgtrain = xgb.DMatrix(train_X, label=train_Y)
'''
cv方法用于进行交叉验证(Cross-validation)
metrics:用于观测的损失函数
as_pandas:以pandas的dataframe格式输出
verbose_eval:是否显示训练进程
seed:随机种子参数
return:每次检验评估的数据，停止时的检验次数即为最佳树个数
'''
cvresult = xgb.cv(xgb_param, xgtrain,           num_boost_round=alg.get_params()['n_estimators'], nfold=5, metrics='auc', early_stopping_rounds=100, verbose_eval=True)
xgb.set_params(n_estimators=cvresult.shape[0])
```

(2)利用网格搜索寻找最佳参数值（较为耗时）
```python
train_X = data.drop(['next_step'], axis=1)
train_Y = data.next_step
param_test = {
        'max_depth': list(range(3, 10, 2)),
        'min_child_weight': list(range(1, 6, 2))
    }
'''
GridSearchCV方法
estimator:待测试模型
param_grid:待测试的参数格网
scoring:评价方式
cv:相当于交叉验证中的nFolds
n_jobs:调用线程个数（一般为cpu个数）
'''
gsearch = GridSearchCV(
    estimator=XGBClassifier(learning_rate=0.01,
    n_estimators=233,
    max_depth=5,
    min_child_weight=3,
    gamma=0,
    subsample=0.9,
    colsample_bytree=0.7,
    objective='binary:logistic',
    scale_pos_weight=9,
    seed=27,
    silent=False,
    max_delta_step=1,
    reg_alpha=225,
    reg_lambda=1),
    param_grid=param_test,
    scoring='roc_auc',
    n_jobs=4,
    iid=False,
    cv=5)
gsearch.fit(train_X, train_Y)
# best_params会给出当前格网中搜索得到的最优参数值
print(gsearch.grid_scores_)
print(gsearch.best_params_)
print(gsearch.best_score_)
```
部分参数介绍：
```python
'''
目标函数
reg:linear线性回归。
reg:logistic逻辑回归。
binary:logistic二分类的逻辑回归问题，输出为概率。
multi:softmax多分类问题，同时需要设置参数num_class
multi:softprob和softmax一样，但是输出的是ndata * nclass的向量，每行数据表示样本所属于每个类别的概率。
'''
objective
# 学习率，降低学习率可以使模型更优，但是速度慢
eta(0-1)
'''
能看懂的就这几个：
rmse:均方误差
error:二分错误率，样本不平衡的时候不能用
merror:多分类错误率
“auc”:roc曲线下面积
'''
eval_metric
# 决策树数量
n_estimators
# 最大树深度，太大容易发生过拟合
max_depth(3-10)
# 最小子节点权重和，越大则树构建过程越保守，不易出现过拟合
min_child_weight(3-10)
# 这个我也看不懂，调一下会有提升
gamma(>0)
# L1正则项系数
reg_alpha(>0)
# L2正则项系数
reg_lambda(>0)
# 样本抽样率
subsample(0.6-1)
# 特征抽样率（列抽样）
colsample_bytree(0.6-1)
# 解决样本不平衡问题，数值设定在负样本/正样本附近进行格网搜索
scale_pos_weight
# 随机种子
seed
# 设为1开启静默模式（无输出）
silent
# 样本不平衡时使用，平衡时为1
max_delta_step(>=1)
```

调参一般过程：
- 设置较大（0.1）的学习率，通过cv求解最优决策树数量
- 网格搜索优化最大深度和最小子节点权重和
- 优化正则项系数和gamma
- 优化样本采样率和特征采样率
- 如果为不平衡样本，优化scale_pos_weight和max_delta_step
- 减小学习率（0.0005-0.001）重新计算n_estimators，使用其他最优参数进行训练得到最终模型

模型结果输出方式
```python
bst = xgb.Booster()
bst.load_model('xgboost.model')
# 特征权重值：特征在模型中用于分类出现的次数
score_weight = bst.get_score('fmap.txt', 'weight')
# 特征产出值：特征用于分割的平均信息增量
score_gain = bst.get_score('fmap.txt', 'gain')
# 使用matplot输出决策树
fig, ax = plt.subplots()
fig.set_size_inches(60, 30)
xgb.plot_tree(bst, ax=ax, num_trees=0, rankdir='LR')
fig.savefig('tree.jpg', dpi=100)
fig.show()
# 使用graphviz绘制决策树，需要安装graphviz软件，输出pdf
xgb.to_graphviz(bst, num_trees=1, rankdir='RL').render()
# 输出特征重要性柱状图
xgb.plot_importance(bst, fmap='fmap.txt')
# 所有的输出防止都可以制定fmap，fmap为特征映射表，每行格式为<feature_id> <feature_name> <feature_type>
plt.show()
```


更多信息参考：[官方参数文档][3]以及[官方API文档][4]

[0]: https://blog.csdn.net/a819825294/article/details/51188740

[1]:http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor

[2]:https://pan.baidu.com/s/1dF2mDbz

[3]:http://xgboost.readthedocs.io/en/latest/parameter.html

[4]:http://xgboost.readthedocs.io/en/latest/python/python_api.html