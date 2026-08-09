#This is my Python Code for Case Studies in Data Science - Individual Task 1 
#This code is for a decision tree classifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

myData=pd.read_csv('student_performance_dataset.csv')
xVars=myData.iloc[:,1:10]
xVars=pd.get_dummies(xVars)
yVar=myData.iloc[:,11]
#student_id is left out as it is not usefull here
#final_exam_score is there is no point predicting a final grade after the final exam has been sat, there is no use in that

xTrain, xTest, yTrain, yTest = train_test_split(xVars, yVar, test_size=0.2)
splitterBest=0
maxDepth=0
accuracyBest=0
splitter = ["best", "random"]
for s in splitter:
    for i in range(1,11):
        clf=DecisionTreeClassifier(max_depth=i,splitter=s)
        clf2=clf.fit(xTrain, yTrain)
        predict=clf2.predict(xTest)
        accuracy=accuracy_score(yTest, predict)
        if accuracy>accuracyBest:
            accuracyBest=accuracy
            splitterBest=s
            maxDepth=i
            #print(accuracy)

clf=DecisionTreeClassifier(max_depth=maxDepth, splitter=splitterBest)
finalModel=clf.fit(xTrain, yTrain)
predict=finalModel.predict(xTest)
finalAccuracy=accuracy_score(yTest, predict)

print('Model is complete')
print(f'Model accuracy: {finalAccuracy}')

#Code is finished, need to fix random so that it is consistant
