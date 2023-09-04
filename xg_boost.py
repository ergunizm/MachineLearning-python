import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
le = LabelEncoder()
y_train = le.fit_transform(y_train) #class column has to start from 0 (we add this because of xgboost)

classifier = XGBClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred = le.inverse_transform(y_pred) # because of the le

cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred)
print(cm, score)

#kfold validation
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print("Accuracy : %{:.2f}".format(accuracies.mean() *100))
print("Standard Deviation : {:.2f}%".format(accuracies.std() *100))