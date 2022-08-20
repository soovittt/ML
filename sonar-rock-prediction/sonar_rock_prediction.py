import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


#read the data from the csv file 
sonar_rock_data = pd.read_csv('sonar_rock_data.csv',header=None)


#data preprocessing 



#split the data into features and target 
features = sonar_rock_data.drop(columns=60,axis=1)
target = sonar_rock_data[60]
# print(features)
# print(target)


#splitting the data into train and test data 
X_train,X_test,Y_train,Y_test = train_test_split(features,target,random_state=2,test_size=0.2)


#create the model and fitting the data
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train,Y_train)


#make a prediction on the train data and calculate the accuracy score
Y_train_pred = logistic_regression.predict(X_train)
Y_train_accuracy_score = accuracy_score(Y_train_pred,Y_train)
print("The train accuracy score is : ",Y_train_accuracy_score)

#make a prediction on the test data and calculate the accuracy score
Y_test_pred = logistic_regression.predict(X_test)
Y_test_accuracy_score = accuracy_score(Y_test_pred,Y_test)
print("The test accuracy score is : ",Y_test_accuracy_score)


#make a predictive model
data = (0.0260,0.0363,0.0136,0.0272,0.0214,0.0338,0.0655,0.1400,0.1843,0.2354,0.2720,0.2442,0.1665,0.0336,0.1302,0.1708,0.2177,0.3175,0.3714,0.4552,0.5700,0.7397,0.8062,0.8837,0.9432,1.0000,0.9375,0.7603,0.7123,0.8358,0.7622,0.4567,0.1715,0.1549,0.1641,0.1869,0.2655,0.1713,0.0959,0.0768,0.0847,0.2076,0.2505,0.1862,0.1439,0.1470,0.0991,0.0041,0.0154,0.0116,0.0181,0.0146,0.0129,0.0047,0.0039,0.0061,0.0040,0.0036,0.0061,0.0115)
data_arr = np.array(data)
reshaped_data_arr = data_arr.reshape(1,-1)
prediction = logistic_regression.predict(reshaped_data_arr)
if(prediction=='M'):
    print("Its a Mine")
elif(prediction=='R'):
    print("Its a Rock")