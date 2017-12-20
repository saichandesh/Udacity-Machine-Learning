#!/usr/bin/python

import sys

from time import time
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

features_train, features_test, labels_train, labels_test = preprocess()

clf = GaussianNB()

t0 = time()

clf.fit(features_train, labels_train)

print "training time:", round(time()-t0, 3), "s"

t0 = time()

clf.predict(features_test)

print "testing time:", round(time()-t0, 3), "s"

print "Accuracy : ",clf.score(features_test,labels_test)
