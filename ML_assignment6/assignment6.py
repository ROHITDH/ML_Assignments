#assignment_6 ML 6/Sep/2020 RDH

#QN:1.Build decision tree model to predict survival based on certain parameters
#	In this file using following columns build a model to predict if person would survive or not,
#		Pclass
#		Sex
#		Age
#		Fare
#	Calculate score of your model

#------------------------------------------------1-----------------------------------------------------------------------
#importing packages related to ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import pickle

#reading data and printing csv (head)
df = pd.read_csv("res/titanic.csv")
print(df.head())

#assigning dependent and independent variable
x = df[['Pclass', 'Sex', 'Age' ,'Fare']] 
y = df[['Survived']] 

#plotting x vs y
plt.scatter(df.iloc[:,0],y)
plt.show()

#encoding sex to int format
le_sex = LabelEncoder()
x['sex_n'] = le_sex.fit_transform(x['Sex'])

#removing older sex column(character) and preprocessing
x = x.drop(['Sex'],axis='columns')
x['Pclass'] = x['Pclass'].fillna(x["Pclass"].median())
x['sex_n'] = x['sex_n'].fillna(x["sex_n"].median())
x['Age'] = x['Age'].fillna(x["Age"].median())
x['Fare'] = x['Fare'].fillna(x["Fare"].median())

#converting to numpy array
x1=x["Pclass"]
x2=x["Age"]
x3=x["Fare"]
x4=x["sex_n"]

x1=x1.to_numpy()
x2=x2.to_numpy()
x3=x3.to_numpy()
x4=x4.to_numpy()
y1=y.to_numpy()

#plotting the converted values
plt.scatter(x1, x2, x3, c= x4)
plt.title("Point observations")
plt.xlabel("Pclass Age, Fare")
plt.ylabel("sex_n")
cbar= plt.colorbar()
cbar.set_label("elevation (m)", labelpad=+1)
plt.show()


#train test split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.20)
model = tree.DecisionTreeClassifier()
model.fit(X_train, Y_train)
with open('model_TitanicSurview_Predict','wb') as file:
    pickle.dump(model,file)
model.score(X_test, Y_test)

#printing accuracy
print("Accuracy: ",model.score(X_test, Y_test))
#ORDER : Pclass, Age, Fare, Sex
#Uncomment below to predict
#model.predict([[3,22.0,7.25,1]])