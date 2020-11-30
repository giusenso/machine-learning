"""
Machine Learning - Homework 1
Author: Giuseppe Sensolini Arra', 1661198
November 2020    
"""

import feature_extraction as fe
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import f1_score
import eli5 as eli
import numpy as np
import matplotlib.pyplot as plt
import graphviz
import math
import json
import time

start = time.time()

# =============================================================================
# EXRACT FEATURES FROM DATASET

print("\n====================\nFEATURE EXTRACTION:\n")
data, train_set, test_set, blind_set = [],[],[],[]
data_set = "../dataset/noduplicatedataset.json"
blind_data = "../dataset/blindtest.json"
print("reading dataset...")

# open dataset
with open(data_set) as f1:
    for line in f1:
        data.append(json.loads(line))

# features extraction
fncts, labels, header, classes = fe.feature_extraction(data)

# divide the dataset in training (80%) and test (20%)  
test_size = int(len(labels)*0.20)
training_size = len(labels)-test_size
train_fncts = fncts[0:len(fncts)-test_size]
train_labels = labels[0:len(fncts)-test_size]
test_fncts = fncts[len(fncts)-test_size:len(fncts)]
test_labels = labels[len(fncts)-test_size:len(fncts)]

# open blindset
with open(blind_data) as f2:
    for line in f2:
        blind_set.append(json.loads(line))

# blind features extraction
blind_fncts = fe.feature_extraction(blind_set, blind=True)

print("train_set size: " + str(len(train_fncts)))
print("test_set size:  " + str(len(test_fncts)))
print("blind_set size: " + str(len(blind_fncts)))

# =============================================================================
# DECISION TREE

print("\n====================\nTRAINING:\n")
print("Building the decision tree... ")
e1 = time.time()
clf = tree.DecisionTreeClassifier(  criterion="entropy",\
                                    splitter = "best",\
                                    max_features = None)
clf = clf.fit(train_fncts, train_labels)
e2 = time.time()

dot_data = tree.export_graphviz(clf, out_file=None, feature_names=header,
                                class_names=classes, filled=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("../output/tree")
print("done.\n-> Tree structure saved in tree.pdf")

importance = clf.feature_importances_
s = "\nfeature importance:\n"
for i in range(0,len(importance)):
    s = s + "--"+header[i]+": "+str(math.floor(importance[i]*10000)/10000)+"\n"
print(s)

# open a (new) file to write
out_file = open("../output/test_output.txt", "w")
out_file.write(s + "\n")

# =============================================================================
# TEST IT

print("====================\nTESTING RESULTS:\n")
e3 = time.time()
pred_labels = clf.predict(test_fncts)

# Error and Accuracy
errors = 0
for i in range(0,test_size):
    if pred_labels[i]!=test_labels[i]: errors+=1
error = errors/test_size
accuracy = 1-error
s = "Error:    " + str(math.floor(error*10000)/10000) + "\nAccuracy: " + str(math.floor(accuracy*10000)/10000)
print(s)
out_file.write(s + "\n")

# F1 Score
f1 = f1_score(test_labels, pred_labels, average="macro")
s = "\nF1 Score: " + str(math.floor(f1*10000)/10000)
print(s)
out_file.write(s + "\n")

# K-FOLD CROSS VALIDATION
k = 8
k_scores = cross_val_score(clf, fncts, labels, cv=k)
k_accuracy = sum(k_scores)/k
k_error = 1-k_accuracy
s = "\nK-Fold cross validation:\n--Error:    " + str(math.floor(k_error*10000)/10000) + "\n--Accuracy: " + str(math.floor(k_accuracy*10000)/10000)
print(s)
out_file.write(s + "\n")

# Plot confusion matrix
titles_options = [("confusion matrix", None),
                  ("normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf, test_fncts, test_labels,
                                 display_labels=classes,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)
    print("\n" + title)
    print(disp.confusion_matrix)
    plt.savefig("../output/" + title + ".png")
    print("-> plot saved in " + title + ".png.")
#plt.show()

e4 = time.time()
fe_time = math.floor((e1-start)*10000)/10000
train_time = math.floor((e2-e1)*10000)/10000
test_time = math.floor((e4-e3)*10000)/10000
s = ("\ntime elapsed: " + 
        str(math.floor((fe_time+train_time+test_time)*10000)/10000) + " s"
        "\n--feature extraction: " + str(fe_time) + " s"
        "\n--model training:     " + str(train_time) + " s"
        "\n--model testing:      " + str(test_time) + " s")
print(s)
out_file.write(s + "\n")

# close file
out_file.close()
print("\n-> results saved on test_output.txt.")

# =============================================================================
# BLIND TEST

print("\n====================\nBLIND TESTING:\n")

# predict 
pred_classes = clf.predict(blind_fncts)

# open a (new) file to write
pred_labels = []
out_file = open("../output/blindtest_output.txt", "w")
for c in pred_classes:
    pred_labels.append(classes[c])
    out_file.write(classes[c])
    out_file.write("\n")
out_file.close()
print("-> blind results saved on blindtest_output.txt.\n")

# close file
out_file.close()
