import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(x_train, y_train)

print(classifier.predict(sc.transform([[30, 87000]])))

y_pred = classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))

cm = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred)
print(cm, score)

# Grid Search to find best model and parameters
parameters = [{'C': [0.25, 0.5, 0.75, 1], 'kernel': ['linear']},
              {'C': [0.25, 0.5, 0.75, 1], 'kernel': ['rbf'], 'gamma' : [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
gs = GridSearchCV(estimator=classifier, param_grid=parameters, 
                  scoring='accuracy', cv=10, n_jobs=-1)
gs.fit(x_train,y_train)
best_accuracy = gs.best_score_
best_parameters = gs.best_params_
print("Best Accuracy : {:.2f}".format(best_accuracy))
print("Best Parameters",best_parameters)