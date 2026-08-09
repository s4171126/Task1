#This is my Python Code for Case Studies in Data Science - Individual Task 1 
#This code is for a decision tree classifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

myData=pd.read_csv('student_performance_dataset.csv')
#print (myData.columns)
#print (myData.iloc[:,1:10])
#student_id is left out as it is not usefull here
#final_exam_score is there is no point predicting a final grade after the final exam has been sat, there is no use in that

xVars=myData.iloc[:,1:10]
yVar=myData.iloc[:,11]

#print(xVars.head(1))
#print(yVar.head(1))

xTrain, xTest, yTrain, yTest = train_test_split(xVars, yVar, test_size=0.2)

splitterBest=0
maxDepth=0
accuracyBest=0

splitter = ["best", "random"]

for s in splitter:
    print(splitter)