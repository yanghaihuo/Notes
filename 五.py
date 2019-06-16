from sklearn.preprocessing import  Imputer,LabelEncoder,OneHotEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn import neighbors
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#导入数据

iris  = load_iris()
data = iris['data']
iris_data = pd.DataFrame(
         data = data,
         columns = ['sepal_length','sepal_width','petal_length','petal_width']
        )
iris_data["Species"] = iris[ 'target']
iris_data = iris_data.loc[iris_data['Species'] != 0,:]

#数据集分割
print(iris_data)
x,y = iris_data.iloc[:,0:-1],iris_data.iloc[:,-1]
# stratify = y 依据标签y，按原数据y中各类比例，分配给train和test，使得train和test中各类数据的比例与原数据集一样
train_data,test_data,train_target,test_target = train_test_split(x,y,test_size = 0.3,stratify = y)
#特征标准化
# 一般情况下，或者严格点说，在监督学习中，我们需要利用训练集数据对测试集数据进行预测。
# 这里隐含了一个假设，就是训练数据和测试数据实际上是同分布的（因此我们才可以使用训练数据集来预测测试数据集），来自于同一个总体。
# 在进行标准化的过程中就将训练集的均值和方差当做是总体的均值和方差，因此对测试集使用训练集的均值和方差进行预处理。
min_max_scaler = preprocessing.MinMaxScaler()
#实例化0-1标准化方法

X_train = min_max_scaler.fit_transform(train_data.values)
X_test  = min_max_scaler.transform(test_data.values)

#模型拟合
model_KNN = neighbors.KNeighborsClassifier()
model_KNN.fit(X_train,train_target)

#预测结果

Pre_label = model_KNN.predict(X_test)

#指标计算
# 1、混淆矩阵输出
metrics.confusion_matrix(test_target,Pre_label)
TP = 14
FN = 1
FP = 2
TN = 13

# 2、分类准确率计算
metrics.accuracy_score(test_target,Pre_label)
Accuracy = (TP + TN)/(TP + TN + FN + FP)
# Accuracy=(14+13)/(14+1+2+13)

# 3、召回率（Recall、或称灵敏性-Sensitivity）
metrics.recall_score(test_target,Pre_label)
Recall = TP/(TP + FN)

# Recall=14/(14+1)

# 4、精确度（Precision）
metrics.precision_score(test_target,Pre_label)
Precision = TP/(TP + FP)

# Precision=14/(14 + 2)

# 5、F1度量
metrics.f1_score(test_target,Pre_label)
# (2*Precision*Recall) / (Precision+Recall)

# 6、ROC曲线与AUC值
# 注意：此实现仅限于二进制分类任务
fpr,tpr,thresholds = metrics.roc_curve(np.array(test_target),Pre_label,pos_label=2)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()
print(metrics.roc_auc_score(test_target.values -1,Pre_label))
# 0.9