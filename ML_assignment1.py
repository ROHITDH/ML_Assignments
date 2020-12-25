#assignment_1 ML 24/Aug/2020 RDH

#1.Create student table having 3 column name,branch,marks
#importing pandas
import pandas as pd
#creating data for the table
data=[('arun','EC',98),('mahesh','CS',100),('dinesh','MECH',99),('rakesh','EC',98)]
#row for the table
row = [0,1,2,3]
#column for the table
coloumn=['name','branch','marks']
#creating the dataFrame/Table
df=pd.DataFrame(data,row,coloumn)
#printing the table
print(df)



#2.Write a code to view studen only one branch
#Viewing of student only "EC" branch
#incrementer variable to keep track of rows
inc = 0
while inc < len(row):
	#checking branch column of each row with "EC"
    if df.loc[inc,'branch'] == 'EC':
    #if matches print the corresponding name 
        print(df.loc[inc,'name'])
    #increment row counter
    inc = inc + 1         



#3.Write a code to view maximum marks first and lowest marks last
print(df.sort_values('marks',ascending=False))
