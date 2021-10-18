import threading
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import time

def calculate(X_train, X_test, y_train, y_test, result, number, seed=1410):
    clf = RandomForestClassifier(random_state=seed)
    clf.fit(X_train, y_train)
    result[number]  = accuracy_score(y_test, clf.predict(X_test))

df = pd.read_csv('egzo.csv')
y = df.iloc[:, -1]
X = df.iloc[:, :-1]

time_cross_validation = 5
result = np.zeros((time_cross_validation))
kf = KFold(time_cross_validation, shuffle=True, random_state=1410)
start = time.perf_counter()
threads = []
for i, (train_index, test_index) in enumerate(kf.split(X)):
    #result[i] = calculate(X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index])
    t = threading.Thread(target=calculate, args=[X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index], result, i])
    t.start()
    threads.append(t)

for thread in threads:
    thread.join()
stop = time.perf_counter()

print(result)
print(stop-start)

