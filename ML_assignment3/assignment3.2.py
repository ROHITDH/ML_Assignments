#assignment_3.2 ML 26/Aug/2020 RDH

#QN:2.In classroom there is Admision_predict.csv. 
#	This file contains admision statics for a firm.
# 	Given this data, you need to build a machine learning
# 	model for student that can help them decide possibality
# 	of addmission chance. Using this predict Chance of Admit 
#	for following candidates,

#S_No GRE TOEFL UR SOP LOR CGPA
#1 	  354  102  2  3.5 4.5 8.2
#2    368  113  1  4   4.5 7.9


#importing packages related to ML
import pandas as pd
from sklearn.linear_model import LinearRegression

#importing package pickle to save model
import pickle

#Linear regression model
model = LinearRegression()

#Reading and printing data from a file
a = pd.read_csv("res/Admission_Predict.csv")
print(a)

#Data Preprocessing: Fill NA values if exists..
a['GRE Score'] = a['GRE Score'].fillna(a['GRE Score'].median())
a['TOEFL Score'] = a['TOEFL Score'].fillna(a['TOEFL Score'].median())
a['University Rating'] = a['University Rating'].fillna(a['University Rating'].median())
a['SOP'] = a['SOP'].fillna(a['SOP'].median())
a['LOR '] = a['LOR '].fillna(a['LOR '].median())
a['CGPA'] = a['CGPA'].fillna(a['CGPA'].median())
a['Chance of Admit '] = a['Chance of Admit '].fillna(a['Chance of Admit '].median())

#y is dependent and x is independent
y=a['Chance of Admit ']
del a['Serial No.'] #serial no is not dependent on any values
del a['Chance of Admit '] #deleting dependent value column
x=a

model.fit(x,y)

#saving the model 
with open('model_Admission_Predict','wb') as file:
    pickle.dump(model,file)


#predicting the chance of admit from the saved model
with open('model_Admission_Predict','rb') as file:
    model = pickle.load(file)

#1
print("chance of admit of given student 1:")
print('%.2f'%model.predict([[354, 102, 2,3.5,4.5,8.2]])[0])
#2
print("chance of admit of given student 2:")
print('%.2f'%model.predict([[368, 113, 1,4,4.5,7.9]])[0])
