'''
直接得到原始数据的对应的序号列表，将类别信息转化成数值信息应用到模型中去
pd.Categorical( list ).codes

mushrooms[column] = pd.factorize(mushrooms[column])[0] # 返回两个值，取第一个
直接得到原始数据的对应的序号列表，将类别信息转化成数值信息应用到模型中去
mushrooms[column] =pd.Categorical(mushrooms[column]).codes


df['CabinCat'] = pd.Categorical.from_array(df.Cabin.fillna('0').apply(lambda x: x[0])).codes    # pd.Categorical.from_array  numpy数组接口
fig, ax = plt.subplots(figsize=(10,5))
sns.countplot(x='CabinCat', hue='Survived',data=df)
plt.show()



# 妇女/儿童 男士标签
child_age = 18
def get_person(passenger):
    age, sex = passenger
    if (age < child_age):
        return 'child'
    elif (sex == 'female'):
        return 'female_adult'
    else:
        return 'male_adult'

df = pd.concat([df, pd.DataFrame(df[['Age', 'Sex']].apply(get_person, axis=1), columns=['person'])],axis=1)
df = pd.concat([df,pd.get_dummies(df['person'])],axis=1)

numpy.ravel() vs numpy.flatten()
两者所要实现的功能是一致的（将多维数组降位一维），两者的区别在于返回拷贝（copy）还是返回视图（view），
numpy.flatten()返回一份拷贝，对拷贝所做的修改不会影响（reflects）原始矩阵，
而numpy.ravel()返回的是视图（view，也颇有几分C/C++引用reference的意味），会影响（reflects）原始矩阵。


经典部分回顾：https://mp.weixin.qq.com/s/gAqxE9MA8VPHzaysYLSjsQ
利用随机森林预测填补缺失值
Age特征缺失值：Age有20%缺失值，缺失值较多，大量删除会减少样本信息，由于它与Cabin不同，这里将利用其它特征进行预测填补Age，也就是拟合未知Age特征值。
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor

classers = ['Fare','Parch','Pclass','SibSp','TitleCat',
            'CabinCat','female','male', 'Embarked', 'FamilySize', 'NameLength','Ticket_Numbers','Ticket_Id']
etr = ExtraTreesRegressor(n_estimators=200,random_state=0)
X_train = df[classers][df['Age'].notnull()]
Y_train = df['Age'][df['Age'].notnull()]
X_test = df[classers][df['Age'].isnull()]

etr.fit(X_train.as_matrix(),np.ravel(Y_train))
age_preds = etr.predict(X_test.as_matrix())
df['Age'][df['Age'].isnull()] = age_preds






X = np.array(new_data.iloc[:, new_data.columns != 'Class']) # 选取特征列数据
y = np.array(new_data.iloc[:, new_data.columns == 'Class']) # 选取类别label






'''
import numpy as np
print(np.linspace(1,10, num=10)) # 等差数列
print(np.logspace(1,10, num=10)) # 等比数列

