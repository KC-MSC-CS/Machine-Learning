import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics

sc = StandardScaler()

df = pd.read_csv('./dataset.csv')
print(df.head())

x = df.iloc[:, [2,3]].values
y = df.iloc[: , 4].values

xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.25, random_state =0)

xtrain = sc.fit_transform(xtrain)
xtest = sc.fit_transform(xtest)

classifier = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2)
classifier.fit(xtrain, ytrain)

ypred = classifier.predict(xtest)

cm = confusion_matrix(ytest, ypred)

print("Confusion matrix: ", cm)
print("Train set Accuracy: ", metrics.accuracy_score(ytrain, classifier.predict(xtrain)))
print("Test set Accuracy: ", metrics.accuracy_score(ytest, ypred))


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()


print("Accuracy:",metrics.accuracy_score(ytest, ypred))
print("Precision:",metrics.precision_score(ytest, ypred))
print("Recall:",metrics.recall_score(ytest, ypred))

y_pred_proba = classifier.predict_proba(xtest)[::,1]
fpr, tpr, _ = metrics.roc_curve(ytest,  y_pred_proba)
auc = metrics.roc_auc_score(ytest, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

#=============================================
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];

for n in range(1,Ks):
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(xtrain,ytrain)
    yhat=neigh.predict(xtest)
    mean_acc[n-1] = metrics.accuracy_score(ytest, yhat)
    std_acc[n-1]=np.std(yhat==ytest)/np.sqrt(yhat.shape[0])

print(mean_acc)

plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()