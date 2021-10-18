import threading
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import time


def calculate(X_train, X_test, y_train, y_test, result, clfs):
    for clfs_names in clfs:
        clfs[clfs_names].fit(X_train, y_train)
    # result[number]  = accuracy_score(y_test, clf.predict(X_test))


clfs = {
    'SVM': SVC()
}

# datasets = ['appendicitis', 'australian', 'balance', 'banknote', 'breastcan', 'breastcancoimbra', 'bupa', 'coil2000',
#             'cryotherapy', 'ecoli4', 'german', 'glass2', 'glass4', 'glass5', 'haberman', 'ionosphere', 'iris', 'liver',
#             'mammographic', 'monk-2', 'phoneme', 'pima', 'popfailures', 'ring', 'sonar', 'soybean', 'spambase',
#             'spectfheart', 'titanic', 'twonorm', 'vowel0', 'waveform', 'wine', 'wisconsin']

datasets = ['appendicitis']

time_cross_validation = 5

def experiment_thread(datasets=datasets, clfs=clfs, time_cross_validation=time_cross_validation):
    result = np.zeros((len(clfs), len(datasets), time_cross_validation))
    kf = KFold(time_cross_validation, shuffle=True, random_state=1410)
    start = time.perf_counter()
    for data_id, dataset in enumerate(datasets):
        df = pd.read_csv(f'datasets/{dataset}.csv')
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
        threads = []
        for i, (train_index, test_index) in enumerate(kf.split(X)):
                t = threading.Thread(target=calculate,
                                     args=[X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index],
                                           result, clfs])
                t.start()
                threads.append(t)

        for thread in threads:
            thread.join()
    stop = time.perf_counter()
    return stop - start


def experiment_normal(datasets=datasets, clfs=clfs, time_cross_validation=time_cross_validation):
    result = np.zeros((len(clfs), len(datasets), time_cross_validation))
    kf = KFold(time_cross_validation, shuffle=True, random_state=1410)
    start = time.perf_counter()
    for data_id, dataset in enumerate(datasets):
        df = pd.read_csv(f'datasets/{dataset}.csv')
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]

        for i, (train_index, test_index) in enumerate(kf.split(X)):
            for clf_id, clf_name in enumerate(clfs):
                clfs[clf_name].fit(X.iloc[train_index], y.iloc[train_index])

    stop = time.perf_counter()
    return stop - start

print(experiment_thread())
print(experiment_normal())
