import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm

myData=pd.read_csv('student-mat.csv', sep=';')
print(myData.columns)
xVars=myData.iloc[:,1:32]
xVars = pd.get_dummies(xVars)
yVar=myData.iloc[:,32]
xTrain, xTest, yTrain, yTest = train_test_split(xVars, yVar, test_size=0.2, random_state=1)
kernelBest=0
CBest=0
accuracyBest=0
kernel=["linear","poly","rbf","sigmoid","precomputed"]
CSet=[1,10,100]

for i in kernel:
    for C in CSet:
        accuracy=0
        print(f'kernel: {i}')
        print(f'C: {C}')
        print('accuracy: ?')
        if accuracy>accuracyBest:
            kernelBest=i
            CBest=C
            accuracyBest=accuracy

#Make final model
finalAccuracy=0
print('Model is complete')
print(f'Model accuracy: {finalAccuracy}')