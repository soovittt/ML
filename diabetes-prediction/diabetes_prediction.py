import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#read the data from the csv file
diabetes_data = pd.read_csv('diabetes.csv')

#split the data into the features and target
features = diabetes_data.drop('Outcome',axis=1)
target = diabetes_data['Outcome']



#standardize the data
standardScaler= StandardScaler()
standardScaler.fit(features)
standerdize_features = standardScaler.fit_transform(features)

#split the data into train and test data
X_train , X_test , Y_train , Y_test = train_test_split(standerdize_features,target,random_state=2,test_size=0.2)



#make the Support vector machine model
model = SVC(kernel='linear')
model.fit(X_train,Y_train)

#predict train data values and calc the accuracy score 
y_train_prediction = model.predict(X_train)
train_accuracy_score = accuracy_score(y_train_prediction,Y_train)
print("The train data accuracy is ",train_accuracy_score)


#predict test data values and calc the accuracy score 
y_test_prediction = model.predict(X_test)
test_accuracy_score = accuracy_score(y_test_prediction,Y_test)
print("The train data accuracy is ",test_accuracy_score)



#make a prediction system
data = (6,148,72,35,0,33.6,0.627,50)
data_arr = np.array(data)
reshaped_arr = data_arr.reshape(1,-1)
prediction = model.predict(reshaped_arr)
if(prediction[0]==0):
    print("Not diabetes")
else:
    print("This is diabetes")