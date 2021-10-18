from threading import Thread
import numpy as np
import pandas as pd
import time

def get_single_file(name, path, results, number):
    results[number] = pd.read_csv(f'{path}/{name}.csv')

def open_and_save_thread(names, path):
    start = time.perf_counter()
    threads = []
    results = [None]*len(names)
    for i, name in enumerate(names):
        t = Thread(target=get_single_file, args=[name, path, results, i])
        t.start()
        threads.append(t)
    for thread in threads:
        thread.join()
    for i, name in enumerate(names):
        results[i].to_csv(f'dataset_copy/{name}.csv', encoding='utf-8', index=False)
    stop = time.perf_counter()
    return stop - start

def open_and_save_normal(names, path):
    start = time.perf_counter()
    threads = []
    results = [None]*len(names)
    for i, name in enumerate(names):
        get_single_file(name, path, results, i)

    for i, name in enumerate(names):
        results[i].to_csv(f'dataset_copy/{name}.csv', encoding='utf-8', index=False)
    stop = time.perf_counter()
    return stop - start

names = ['appendicitis', 'australian', 'balance', 'banknote', 'breastcan', 'breastcancoimbra', 'bupa', 'coil2000',
         'cryotherapy', 'ecoli4', 'german', 'glass2', 'glass4', 'glass5', 'haberman', 'ionosphere', 'iris', 'liver',
         'mammographic', 'monk-2', 'phoneme','pima','popfailures','ring','sonar','soybean','spambase','spectfheart',
         'titanic','twonorm','vowel0','waveform','wine', 'wisconsin']

print(open_and_save_thread(names, 'datasets'))
print(open_and_save_normal(names, 'datasets'))

