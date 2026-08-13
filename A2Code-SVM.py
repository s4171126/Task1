import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

myData=pd.read_csv('student-mat.csv', sep=';')
LG=[]
for i in myData.iloc[:,32]:
    if int(i)<=4:
        LG.append('F')
    elif int(i)<=8:
        LG.append('D')
    elif int(i)<=12:
        LG.append('C')
    elif int(i)<=16:
        LG.append('B')
    elif int(i)<=20:
        LG.append('A')
myData['LG']=LG

xVars=myData.iloc[:,1:32]
xVars = pd.get_dummies(xVars)
yVar=myData.iloc[:,33]
xTrain, xTest, yTrain, yTest = train_test_split(xVars, yVar, test_size=0.2, random_state=1)
kernelBest=0
CBest=0
accuracyBest=0
kernel=["linear","poly","rbf","sigmoid"]
CSet=[1,10,100]

for i in kernel:
    for CVal in CSet:
        clf= SVC(kernel=i, C=CVal, random_state=1)
        clf2=clf.fit(xTrain, yTrain)
        predict=clf2.predict(xTest)
        accuracy=accuracy_score(yTest, predict)
        if accuracy>accuracyBest:
                kernelBest=i
                CBest=CVal
                accuracyBest=accuracy

clf=SVC(kernel=kernelBest, C=CBest, random_state=1)
finalModel=clf.fit(xTrain, yTrain)
predict = finalModel.predict(xTest)
finalAccuracy = accuracy_score(yTest,predict)
print('Model is complete')
print(f'Model accuracy: {finalAccuracy}')