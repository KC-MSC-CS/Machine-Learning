import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('./Social_Network_Ads.csv')
print('Dataset :', data.shape)
print(data.info())

Gender = {'Male': 1, 'Female': 0}

# traversing through dataframe
# Gender column and writing
# values where key matches
data.Gender = [Gender[item] for item in data.Gender]
print(data)


Y = data['Purchased']
X = data.drop(columns=['Purchased'])
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=9)
print('X train shape: ', X_train.shape)
print('Y train shape: ', Y_train.shape)
print('X test shape: ', X_test.shape)
print('Y test shape: ', Y_test.shape)


# We define the model
rfcla = RandomForestClassifier(n_estimators=10, random_state=9, n_jobs=-1)

# We train model
rfcla.fit(X_train, Y_train)

# We predict target values
Y_predict5 = rfcla.predict(X_test)

# The confusion matrix
rfcla_cm = confusion_matrix(Y_test, Y_predict5)
rfcla_cm

f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(rfcla_cm, annot=True, linewidth=0.7,
            linecolor='black', fmt='g', ax=ax, cmap="BuPu")
plt.title('Random Forest Classification Confusion Matrix')
plt.xlabel('Y predict')
plt.ylabel('Y test')
plt.show()

# Instantiate dt
dt = DecisionTreeClassifier(random_state=1)
# Instantiate bc
bc = BaggingClassifier(base_estimator=dt, n_estimators=150, random_state=1)
# Fit bc to the training set
bc.fit(X_train, Y_train)
# Predict test set labels
y_pred = bc.predict(X_test)

bgcla_cm = confusion_matrix(Y_test, y_pred)
print(bgcla_cm)

# Evaluate acc_test
acc_test = accuracy_score(Y_test, y_pred)
print('Test set accuracy of bc: {:.2f}'.format(acc_test))

# AdaBoostClassifier
ada_boost_clf = AdaBoostClassifier(n_estimators=50)
ada_boost_clf.fit(X_train, Y_train)
a = ada_boost_clf.predict(X_test)

bocla_cm = confusion_matrix(Y_test, a)
print(bocla_cm)

acc_test = accuracy_score(Y_test, a)
print('Test set accuracy of bc: {:.2f}'.format(acc_test))
