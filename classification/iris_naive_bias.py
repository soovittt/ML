'''
This is the iris dataset classification using the Gaussian naive bias classifiers.
'''

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns


#load the data and convert it into DataFrame
iris_data = load_iris()
iris_data_dataframe = pd.DataFrame(iris_data.data,columns=iris_data.feature_names)
iris_data_dataframe['target'] = iris_data.target


#seeing the correlation between the features
graph = sns.heatmap(iris_data_dataframe.drop('target',axis=1).corr(),cmap='Blues',annot=True,cbar=True)
# plt.show()


#split the data into features and target
features = iris_data_dataframe.drop('target',axis=1)
target = iris_data_dataframe['target']


#split the data into testing and training data
X_train,X_test,Y_train,Y_test  = train_test_split(features,target,random_state=2,test_size=0.2)


#create a Gaussian Naive Bias Classifier
gnb = GaussianNB()
gnb.fit(X_train.values,Y_train.values)
y_train_pred = gnb.predict(X_train.values)
train_accuracy_score = accuracy_score(Y_train.values,y_train_pred)
# print(f"accuracy score of training data : {train_accuracy_score}")
# print(f"accuracy percentage of training data : {train_accuracy_score*100}")


#test the model on the testing data
y_test_pred = gnb.predict(X_test.values)
test_accuracy_score = accuracy_score(Y_test.values,y_test_pred)
# print(f"accuracy score of testing data : {test_accuracy_score}")
# print(f"accuracy percentage of testing data : {test_accuracy_score*100}")




#make a predict system
dataset = np.array([[6.5,3.0,5.2,2.0]])
prediction = gnb.predict(dataset)
if(prediction==0):
    print("setosa")
elif(prediction==1):
    print("versicolor")
elif(prediction==2):
    print("virginica")