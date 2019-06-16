# from imblearn.over_sampling import SMOTE
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
#
# print('训练集的交易记录条数：',X_train.shape[0])
# print('测试集的交易记录条数：',X_test.shape[0])
# print('交易记录总数：',X_train.shape[0] + X_test.shape[0])
# print('上采样前，类别为‘1’的共有{}个，类别为‘0’的共有{}个。'.format(sum(y_train==1),sum(y_train==0)))
# print('------------------------')
#
# # 对训练集进行上采样处理
# smote = SMOTE(random_state=2)
# X_train_os,y_train_os = smote.fit_sample(X_train, y_train.ravel()) # ravel(): change the shape of y to (n_samples, )
#
# print('上采样后，训练集的交易记录条数：', len(X_train_os))
# print('其中，训练集X的shape:',X_train_os.shape,'，y的shape:',y_train_os.shape)
# print('交易记录总数：',X_train_os.shape[0] + X_test.shape[0])
# print('上采样后，类别为‘1’的共有{}个，类别为‘0’的共有{}个。'.format(sum(y_train_os==1),sum(y_train_os==0)))
#
#
# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix,roc_curve, auc, recall_score, classification_report
#
# # 定义正则化权重参数，用以控制过拟合
# paramaters = {'C':np.linspace(1,10, num=10)} # generate sequnce: start = 1, stop = 10
# paramaters
# # C_param_range = [0.01,0.1,1,10,100]
#
# lr = LogisticRegression()
# # 5 folds, 3 jobs run in parallel
# lr_clf = GridSearchCV(lr, paramaters, cv=5, n_jobs=3, verbose=5)
# lr_clf.fit(X_train_os, y_train_os.ravel())
#
# GridSearchCV(cv=5, error_score='raise-deprecating',
#        estimator=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#           intercept_scaling=1, max_iter=100, multi_class='warn',
#           n_jobs=None, penalty='l2', random_state=None, solver='warn',
#           tol=0.0001, verbose=0, warm_start=False),
#        fit_params=None, iid='warn', n_jobs=3,
#        param_grid={'C': array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])},
#        pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
#        scoring=None, verbose=5)
#
# print('最好的参数：',lr_clf.best_params_)
#
# lr1 = LogisticRegression(C=4, penalty='l1',verbose=5)
# lr1.fit(X_train_os, y_train_os.ravel())
#
# LogisticRegression(C=4, class_weight=None, dual=False, fit_intercept=True,
#           intercept_scaling=1, max_iter=100, multi_class='warn',
#           n_jobs=None, penalty='l1', random_state=None, solver='warn',
#           tol=0.0001, verbose=5, warm_start=False)
#
#
#
#
#
# data_train.describe(include=['O'])
#
# # 将oject数据转化为int类型
# for feature in data.columns:
#     if data[feature].dtype == 'object':
#         data[feature] = pd.Categorical(data[feature]).codes # codes	这个分类的分类代码
#
#
# '''
# '''
# 选取特征数据与类别数据
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
#
# X_df = data.iloc[:,data.columns != 'income_bracket']
# y_df = data.iloc[:,data.columns == 'income_bracket']
#
# X = np.array(X_df)
# y = np.array(y_df)
#
#
# 特征重要性评估
# 在这里我们使用DecisionTreesClassifier来判断特征变量的重要性
#
# from sklearn.tree import DecisionTreeClassifier
# # from sklearn.decomposition import PCA
#
# # fit an Extra Tree model to the data
# tree = DecisionTreeClassifier(random_state=0)
# tree.fit(X, y)
#
# # 显示每个属性的相对重要性得分
# relval = tree.feature_importances_
#
#
#
# 递归特征消除 (RFE)
# 选取10个重要特征
#
# from sklearn.feature_selection import RFE
#
# # 使用决策树作为模型
# lr = DecisionTreeClassifier()
# names = X_df.columns.tolist()
#
# # 将所有特征排序
# selector = RFE(lr, n_features_to_select = 10)
# selector.fit(X,y.ravel())
#
# print("排序后的特征：",sorted(zip(map(lambda x:round(x,4), selector.ranking_), names)))
#
# # 得到新的dataframe
# X_df_new = X_df.iloc[:, selector.get_support(indices = False)]
# X_df_new.columns
# '''
#
#
# data_int=data.loc[:,['age','fnlwgt','capital.loss','hours.per.week']]
# f,ax=plt.subplots(figsize=(15,15))
# sns.heatmap(data_int.corr(),annot=True, linewidths=.5, fmt= '.1f',ax=ax)
#
#
# len(list(set(basicwords)))
#
# '''
# '''


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets  # sklearn即scikit-learn库
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()  # 方便起见，直接使用sklearn中内置的鸢尾花数据集
X = iris.data[:, :2]  # 为方便可视化，仅取2个特征
y = iris.target

# 展示下数据集中的数据分布
plt.scatter(X[y == 0, 0], X[y == 0, 1])
plt.scatter(X[y == 1, 0], X[y == 1, 1])
plt.scatter(X[y == 2, 0], X[y == 2, 1])
plt.show()

# 为了检测模型的准确率，防止模型在训练集中过拟合，将数据集随机分为训练数据集和测试数据集
X_train, X_test, y_train, y_test = train_test_split(X,  # 样本值
                                                    y,  # 样本对应标签
                                                    random_state=666  # 为每次
                                                    # 运行都得到相同的结果，种了颗随机种子
                                                    )

# 实例化一个kNN模型
knn_clf = KNeighborsClassifier()
# 将KNN模型在训练数据集上进行训练
knn_clf.fit(X_train, y_train)
# 在测试数据集上检测下模型的准确度
accuracy = knn_clf.score(X_test, y_test)
print("Accuracy: ", accuracy)

# 再实例化一个kNN模型
knn_clf2 = KNeighborsClassifier(n_neighbors=6, weights='distance', p=2)
# 将该KNN模型在训练数据集上进行训练
knn_clf2.fit(X_train, y_train)
# 在测试数据集上检测下模型的准确度
accuracy = knn_clf2.score(X_test, y_test)
# 打印准确率
print("Accuracy: ", accuracy)

best_k = -1
best_p = -1
best_accuracy = 0

for k in range(3, 10):
    for p in range(1, 11):
        # 实例化一个kNN模型, 为加快运算速度使n_jobs=-1(使用CPU所有核运算)
        knn_clf2 = KNeighborsClassifier(n_neighbors=k, weights='distance', p=p, n_jobs=-1)
        # 将该KNN模型在训练数据集上进行训练
        knn_clf2.fit(X_train, y_train)
        # 在测试数据集上检测下模型的准确度
        accuracy = knn_clf2.score(X_test, y_test)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
            best_p = p

# 打印最佳参数值
print("最佳k值: %d, 最佳p值 : %d, 最高准确度: %f" % (best_k, best_p, best_accuracy))