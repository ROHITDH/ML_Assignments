#assignment_5 ML 28/Aug/2020 RDH

#QN:1.Download employee retention dataset from here: 
#	https://www.kaggle.com/giripujar/hr-analytics. OR you can choose any data set.
#	1.Now do some exploratory data analysis to figure out which variables 
#		have direct and clear impact on employee retention (i.e. whether they 
#		leave the company or continue to work)
#	2.Plot bar charts showing impact of employee salaries on retention
#	3.Plot bar charts showing corelation between department and employee retention
#	4.Now build logistic regression model using variables that were narrowed
# 		down in step 1. Measure the accuracy of the model



#------------------------------------------------1-----------------------------------------------------------------------
#importing packages related to ML
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

#importing package pickle to save model
import pickle


#LogisticRegression model
model = LogisticRegression()

#Reading and printing data from a file
df = pd.read_csv("res/datasets_11142_15488_HR_comma_sep.csv")


#Using pandas to create dummy variables

dummies_1 = pd.get_dummies(df['salary'])


#adding dummies to data frame and ct=reating new dataframe
df_dummies= pd.concat([df,dummies_1],axis='columns')




#---------------------Plotting and comparing all the graphs--------------------------------------
#Dummy variables meet the assumption of logistic by definition
#plotting the values of df

#1.satisfaction_level vs left
plt.scatter(df_dummies['satisfaction_level'],df_dummies['left'])
plt.xlabel('satisfaction_level', fontsize=18)
plt.ylabel('left', fontsize=18)
plt.show()

#2.last_evaluation vs left
plt.scatter(df_dummies['last_evaluation'],df_dummies['left'])
plt.xlabel('last_evaluation', fontsize=18)
plt.ylabel('left', fontsize=18)
plt.show()

#3.number_project vs left
plt.scatter(df_dummies['number_project'],df_dummies['left'])
plt.xlabel('number_project', fontsize=18)
plt.ylabel('left', fontsize=18)
plt.show()

#4.average_montly_hours vs left
plt.scatter(df_dummies['average_montly_hours'],df_dummies['left'])
plt.xlabel('average_montly_hours', fontsize=18)
plt.ylabel('left', fontsize=18)
plt.show()

#5.time_spend_company vs left
plt.scatter(df_dummies['time_spend_company'],df_dummies['left'])
plt.xlabel('time_spend_company', fontsize=18)
plt.ylabel('left', fontsize=18)
plt.show()

#6.time_spend_company vs left
plt.scatter(df_dummies['Work_accident'],df_dummies['left'])
plt.xlabel('Work_accident', fontsize=18)
plt.ylabel('left', fontsize=18)
plt.show()

#7.Work_accident vs left
plt.scatter(df_dummies['promotion_last_5years'],df_dummies['left'])
plt.xlabel('promotion_last_5years', fontsize=18)
plt.ylabel('left', fontsize=18)
plt.show()

#8.Department vs left
plt.scatter(df_dummies['Department'],df_dummies['left'])
plt.xlabel('Department', fontsize=18)
plt.ylabel('left', fontsize=18)
plt.show()

#9.salary vs left
plt.scatter(df_dummies['salary'],df_dummies['left'])
plt.xlabel('salary', fontsize=18)
plt.ylabel('left', fontsize=18)
plt.show()


print("""As per the graph, the following have direct and more relevent impact on employ retention

\t1.satisfaction_level\n	
\t2.last_evaluation\n
\t3.number_project\n
\t4.average_montly_hours\n
\t5.time_spend_company\n
\t6.salary\n
\t7.work accident\n
\t8.promotion\n

""")


#------------------------------------------------2-----------------------------------------------------------------------

#---------------------------------impact of employ salary on retension--------------------------------
sn.set(style="whitegrid")
sn.barplot(x=df['salary'], y=df['left'])
plt.show()


#------------------------------------------------3-----------------------------------------------------------------------
#---------------------------------impact of employ Department on retension--------------------------------
sn.set(style="whitegrid")
sn.barplot(x=df['Department'], y=df['left'])
plt.show()



#------------------------------------------------4-----------------------------------------------------------------------

#deleting character content row of dependent variable from dataframe(copy)
del df_dummies['Department'] #not cause anything to left

del df_dummies['salary'] #dummies of salary created


df_dummies_copy = df_dummies
#Dummy Variable Trap
#y is dependent and x is independent
y = df_dummies['left']

del df_dummies['left']
x = df_dummies

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.20)

X_scaled = preprocessing.scale(X_train)
Y_scaled = preprocessing.scale(Y_train)

model.max_iter=1000

#fitting values to the model
model.fit(X_train,Y_train)

#printing it's coeff of determination
print("coefficient of determination R^2 of model is : ",model.score(X_test, Y_test),'\n')
    
#saving the model 
with open('model_HR','wb') as file:
	pickle.dump(model,file)

print("MODEL CREATED\n\n")

#load the saved model
with open('model_HR','rb') as file:
	model = pickle.load(file)

#predicting for test case values
predicted = model.predict(X_test)

#creating confussion matrix to check accuracy
results = confusion_matrix(Y_test,predicted)
print("Confustion matrix \n",results)

#plotting the confussion matrix
print("\nConfustion matrix PLOT \n")

plt.figure(figsize = (10,7))
sn.heatmap(results, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')



print("""From conussion matrix, we can conclude that, since the data are scattered more,\n
It's not more accurate prediction, and we can observe more deviation in confussion matrix graph""")

#for scattering data uncomment it

#data = np.array(df_dummies_copy)
#num_train = int(.9*len(data))
#x_train, y_train = data[:num_train, :-1], data[:num_train,-1]
#x_test,y_test = data[num_train:, :-1], data[num_train:,-1]
#plt.scatter(x_train[:,2],x_train[:,3], c=y_train, alpha = 0.5)

