from sklearn import datasets, metrics, ensemble, tree, svm, neighbors, dummy
import xgboost
import lightgbm as lgb
import time
import numpy as np


mnist = datasets.fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data / 255, mnist.target
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
print(X.shape, y.shape)


def run(model, name):
    start = time.time()
    model.fit(X_train, y_train)
    print(f'[{name}] {time.time() - start}')
    y_pred = model.predict(X_test)
    print(metrics.classification_report(y_test, y_pred))


n_ens = 100
run(dummy.DummyClassifier(strategy='stratified'), 'dummy')
run(tree.DecisionTreeClassifier(), 'decision tree')
run(ensemble.RandomForestClassifier(n_estimators=n_ens), 'random forest')
run(ensemble.ExtraTreesClassifier(n_estimators=n_ens), 'extra tree')
run(xgboost.XGBClassifier(n_estimators=n_ens), 'xgboost')
run(lgb.LGBMClassifier(objective='multiclass', num_class=10), 'lightgbm')
run(svm.SVC(), 'svc')
run(neighbors.KNeighborsClassifier(), 'knn')


''' output
(70000, 784) (70000,)
[dummy] 0.03218722343444824
              precision    recall  f1-score   support

           0       0.10      0.10      0.10       980
           1       0.11      0.12      0.12      1135
           2       0.10      0.10      0.10      1032
           3       0.10      0.10      0.10      1010
           4       0.11      0.10      0.11       982
           5       0.08      0.08      0.08       892
           6       0.10      0.10      0.10       958
           7       0.12      0.12      0.12      1028
           8       0.11      0.11      0.11       974
           9       0.11      0.11      0.11      1009

    accuracy                           0.11     10000
   macro avg       0.11      0.10      0.11     10000
weighted avg       0.11      0.11      0.11     10000

[decision tree] 18.67825198173523
              precision    recall  f1-score   support

           0       0.92      0.94      0.93       980
           1       0.96      0.96      0.96      1135
           2       0.87      0.84      0.86      1032
           3       0.82      0.86      0.84      1010
           4       0.87      0.87      0.87       982
           5       0.84      0.83      0.84       892
           6       0.90      0.89      0.89       958
           7       0.90      0.90      0.90      1028
           8       0.83      0.81      0.82       974
           9       0.85      0.84      0.84      1009

    accuracy                           0.88     10000
   macro avg       0.88      0.88      0.88     10000
weighted avg       0.88      0.88      0.88     10000

[random forest] 42.59108304977417
              precision    recall  f1-score   support

           0       0.97      0.99      0.98       980
           1       0.99      0.99      0.99      1135
           2       0.96      0.97      0.96      1032
           3       0.96      0.96      0.96      1010
           4       0.97      0.97      0.97       982
           5       0.97      0.95      0.96       892
           6       0.97      0.98      0.98       958
           7       0.97      0.96      0.97      1028
           8       0.96      0.95      0.96       974
           9       0.96      0.95      0.95      1009

    accuracy                           0.97     10000
   macro avg       0.97      0.97      0.97     10000
weighted avg       0.97      0.97      0.97     10000

[extra tree] 35.90268397331238
              precision    recall  f1-score   support

           0       0.98      0.99      0.98       980
           1       0.99      0.99      0.99      1135
           2       0.97      0.97      0.97      1032
           3       0.97      0.97      0.97      1010
           4       0.98      0.97      0.97       982
           5       0.98      0.97      0.97       892
           6       0.98      0.98      0.98       958
           7       0.97      0.96      0.97      1028
           8       0.97      0.97      0.97       974
           9       0.96      0.96      0.96      1009

    accuracy                           0.97     10000
   macro avg       0.97      0.97      0.97     10000
weighted avg       0.97      0.97      0.97     10000

[xgboost] 1130.1642940044403
              precision    recall  f1-score   support

           0       0.97      0.99      0.98       980
           1       0.99      0.99      0.99      1135
           2       0.97      0.97      0.97      1032
           3       0.97      0.98      0.98      1010
           4       0.98      0.97      0.98       982
           5       0.98      0.97      0.98       892
           6       0.98      0.98      0.98       958
           7       0.98      0.97      0.97      1028
           8       0.97      0.97      0.97       974
           9       0.97      0.97      0.97      1009

    accuracy                           0.98     10000
   macro avg       0.98      0.98      0.98     10000
weighted avg       0.98      0.98      0.98     10000

[lightgbm] 59.470603942871094
              precision    recall  f1-score   support

           0       0.98      0.99      0.98       980
           1       0.99      0.99      0.99      1135
           2       0.97      0.98      0.97      1032
           3       0.97      0.98      0.98      1010
           4       0.98      0.98      0.98       982
           5       0.98      0.97      0.97       892
           6       0.98      0.98      0.98       958
           7       0.98      0.97      0.97      1028
           8       0.96      0.98      0.97       974
           9       0.97      0.97      0.97      1009

    accuracy                           0.98     10000
   macro avg       0.98      0.98      0.98     10000
weighted avg       0.98      0.98      0.98     10000

[svc] 231.19305205345154
              precision    recall  f1-score   support

           0       0.98      0.99      0.99       980
           1       0.99      0.99      0.99      1135
           2       0.98      0.97      0.98      1032
           3       0.97      0.99      0.98      1010
           4       0.98      0.98      0.98       982
           5       0.99      0.98      0.98       892
           6       0.99      0.99      0.99       958
           7       0.98      0.97      0.97      1028
           8       0.97      0.98      0.97       974
           9       0.97      0.96      0.97      1009

    accuracy                           0.98     10000
   macro avg       0.98      0.98      0.98     10000
weighted avg       0.98      0.98      0.98     10000

[knn] 0.12223625183105469
              precision    recall  f1-score   support

           0       0.96      0.99      0.98       980
           1       0.95      1.00      0.98      1135
           2       0.98      0.96      0.97      1032
           3       0.96      0.97      0.97      1010
           4       0.98      0.96      0.97       982
           5       0.97      0.97      0.97       892
           6       0.98      0.99      0.98       958
           7       0.96      0.96      0.96      1028
           8       0.99      0.94      0.96       974
           9       0.96      0.95      0.95      1009

    accuracy                           0.97     10000
   macro avg       0.97      0.97      0.97     10000
weighted avg       0.97      0.97      0.97     10000
'''
