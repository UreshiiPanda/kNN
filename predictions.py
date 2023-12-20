'''
USAGE:
    
# take your best model and use it to produce income.test.predicted
# python3 predictions.py > income.test.predicted
# then validate your results:      cat income.test.predicted | python3 validate.py

'''



import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances
from knn import KNNClassifier 

import sys
import os


# take in the input data
test_data = pd.read_csv("income.test.blind", sep=", ", names=["age", "sector", "edu", "marriage", "occupation", "race", "sex", "hours", "country"])
train_data = pd.read_csv("income.train.txt.5k", sep=", ", names=["age", "sector", "edu", "marriage", "occupation", "race", "sex", "hours", "country", "target"])
dev_data = pd.read_csv("income.dev.txt", sep=", ", names=["age", "sector", "edu", "marriage", "occupation", "race", "sex", "hours", "country", "target"])

feature_train = train_data.iloc[:, :9]
target_train = train_data["target"]

feature_dev = dev_data.iloc[:, :9]
target_dev = dev_data["target"]

test_features = test_data.iloc[:, :]



# now we scale/normalize the numerical fields and setup encoders
num_processor = MinMaxScaler(feature_range=(0, 2))
cat_processor = OneHotEncoder(sparse_output=False, handle_unknown='ignore')



# fit the pre-processor to the dataset and transform it
preprocessor = ColumnTransformer([
    ('num1', num_processor, ['age']),
    ('cat1', cat_processor, ['sector']),
    ('cat2', cat_processor, ['edu']),
    ('cat3', cat_processor, ['marriage']),
    ('cat4', cat_processor, ['occupation']),
    ('cat5', cat_processor, ['race']),
    ('cat6', cat_processor, ['sex']),
    ('num2', num_processor, ['hours']),
    ('cat7', cat_processor, ['country']),
])


# pre-process the data
preprocessor.fit(feature_train)
processed_train_data = preprocessor.transform(feature_train)    

processed_dev_data = preprocessor.transform(feature_dev)    

processed_test_data = preprocessor.transform(test_features)




# encode the TRAINING targets into 0/1 values
train_y = target_train.values
y_train = []
for target in train_y:
    if target == '>50K': y_train.append(1)
    else: y_train.append(0)
y_train = np.array(y_train)



# encode the DEV targets into 0/1 values
dev_y = target_dev.values
y_dev = []
for target in dev_y:
    if target == '>50K': y_dev.append(1)
    else: y_dev.append(0)
y_dev = np.array(y_dev)


dev_accuracies = []

## run on DEV data
#for i in range(100):
#    if not i % 2: 
#        continue
#    else:
#        kenene = KNNClassifier(k=i)
#        kenene.fit(processed_train_data, y_train)
#        
#        start = time.time()   
#        y_dev_pred = kenene.predict_euclid(processed_dev_data) 
#        end = time.time()
#
#        print("k value is:  ", i)
#        print(f"time elapsed: {end - start}")
#
#        pos = 0
#        for val in y_dev_pred:
#            if val: pos += 1
#
#        print(f"percentage predicted pos: {pos}/1000 = {pos/1000}")
#        test_accuracy = accuracy_score(y_dev, y_dev_pred)     
#        print(f"Euclid DEV Accuracy: {test_accuracy}")
#        print()
#
#        dev_accuracy = accuracy_score(y_dev, y_dev_pred)            # calc the best DEV accuracy
#        dev_accuracies.append( (dev_accuracy, i) )
#
#
#
#
#print(f"dev EUCLID accuracies arr: {dev_accuracies}")
#print()
#print(f"max dev EUCLID accuracy: {max(dev_accuracies)}")
#print()
#
#
#print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
#print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
#print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
#
#
#for i in range(100):
#    if not i % 2: 
#        continue
#    else:
#        kenene = KNNClassifier(k=i)
#        kenene.fit(processed_train_data, y_train)
#        
#        start = time.time() 
#        y_dev_pred = kenene.predict_manhat(processed_dev_data) 
#        end = time.time()
#
#        print("k value is:  ", i)
#        print(f"time elapsed: {end - start}")
#        pos = 0
#
#        for val in y_dev_pred:
#            if val: pos += 1
#
#        print(f"percentage predicted pos: {pos}/1000 = {pos/1000}")
#        test_accuracy = accuracy_score(y_dev, y_dev_pred)      
#        print(f"Manhat DEV Accuracy: {test_accuracy}")
#        print()
#
#        dev_accuracy = accuracy_score(y_dev, y_dev_pred)            # calc the best DEV accuracy
#        dev_accuracies.append( (dev_accuracy, i) )
#
#
#print(f"dev MANHAT accuracies arr: {dev_accuracies}")
#print()
#print(f"max dev MANHAT accuracy: {max(dev_accuracies)}")
#print()


#print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
#print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
#print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")


# 
# 
# # run on TESTING data
# for i in range(100):
#     if not i % 2: 
#         continue
#     else:
#         kenene = KNNClassifier(k=i)
#         kenene.fit(processed_train_data, y_train)
#         
#         start = time.time()   
#         y_test_pred = kenene.predict_euclid(processed_test_data)      
#         end = time.time()
# 
#         print("k value is:  ", i)
#         print(f"time elapsed: {end - start}")
# 
#         pos = 0
#         for val in y_test_pred:
#             if val: pos += 1
# 
#         print(f"percentage EUCLID predicted pos: {pos}/1000 = {pos/1000}")
# 
# print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
# print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
# print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
# 
# 
# for i in range(100):
#     if not i % 2: 
#         continue
#     else:
#         kenene = KNNClassifier(k=i)
#         kenene.fit(processed_train_data, y_train)
#         
#         start = time.time() 
#         y_test_pred = kenene.predict_manhat(processed_test_data) 
#         end = time.time()
# 
#         print("k value is:  ", i)
#         print(f"time elapsed: {end - start}")
#         pos = 0
# 
#         for val in y_test_pred:
#             if val: pos += 1
# 
#         print(f"percentage MANHAT predicted pos: {pos}/1000 = {pos/1000}")
# 
# print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
# print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")
# print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ")




kenene = KNNClassifier(k=41)
kenene.fit(processed_train_data, y_train)
y_test_pred = kenene.predict_euclid(processed_test_data)      
# y_test_pred = kenene.predict_manhat(processed_test_data)      


with open("income.test.blind", "r") as test_inputs:
    inputs = [line.strip().split(", ") for line in test_inputs]

for i in range(1000):
    if y_test_pred[i] == 0: 
        inputs[i].append('<=50K')
    else:
        inputs[i].append('>50K')

for line in inputs:
    output = ", ".join(line)
    print(output)
