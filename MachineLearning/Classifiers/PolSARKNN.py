import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

dataset=pd.read_csv('D:/28_GitHub/Land Cover Mapping/MachineLearning/TrainDataset/PolSARData.csv')

test_dataset1=pd.read_csv("D:/28_GitHub/Land Cover Mapping/MachineLearning/TestDataset/Del78.csv")
test_dataset2=pd.read_csv("D:/28_GitHub/Land Cover Mapping/MachineLearning/TestDataset/Del256.csv")
test_dataset3=pd.read_csv("D:/28_GitHub/Land Cover Mapping/MachineLearning/TestDataset/SF116.csv")
test_dataset4=pd.read_csv("D:/28_GitHub/Land Cover Mapping/MachineLearning/TestDataset/SF307.csv")

Test1=test_dataset1.iloc[:,:]
Test2=test_dataset2.iloc[:,:]
Test3=test_dataset3.iloc[:,:]
Test4=test_dataset4.iloc[:,:]

Test1=np.array(Test1)
Test2= np.array(Test2)
Test3=np.array(Test3)
Test4=np.array(Test4)

X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]
print(X)
print(y)
X=np.array(X)
y=np.array(y)
print(X)
print(y)

#X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

classifier=KNeighborsClassifier(n_neighbors=21, metric='minkowski',p=2)
classifier.fit(X,y)



y_pred1=classifier.predict(Test1)
y_pred2=classifier.predict(Test2)
y_pred3=classifier.predict(Test3)
y_pred4=classifier.predict(Test4)

z1=pd.DataFrame(y_pred1)
z2=pd.DataFrame(y_pred2)
z3=pd.DataFrame(y_pred3)
z4=pd.DataFrame(y_pred4)
z1.to_csv("D:/28_GitHub/Land Cover Mapping/MachineLearning/TestResults/Del78_KNNResults.csv")
z2.to_csv("D:/28_GitHub/Land Cover Mapping/MachineLearning/TestResults/Del256_KNNResults.csv")
z3.to_csv("D:/28_GitHub/Land Cover Mapping/MachineLearning/TestResults/SF116_KNNResults.csv")
z4.to_csv("D:/28_GitHub/Land Cover Mapping/MachineLearning/TestResults/SF307_KNNResults.csv")
#y_pred=(y_pred>0.5)
