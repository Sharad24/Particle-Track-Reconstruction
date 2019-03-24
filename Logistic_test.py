import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import os
import pandas as pd

s_Train = np.load('code/train10events.npy')
clf = LogisticRegression(multi_class='ovr')
clf = GradientBoostingClassifier(max_depth=10)
rows = 20000
x_train = s_Train[:rows, :-1]
y_train = s_Train[:rows, -1:].reshape(s_Train[:rows, :-1].shape[0])

clf.fit(x_train, y_train)

test_start = 100000
test_end = 110000
x_test = s_Train[test_start:test_end, :-1]
y_test = np.array(list(s_Train[test_start:test_end, -1:]))

print(clf.score(x_test, y_test))

clf = LogisticRegression()
clf = GradientBoostingClassifier(max_depth=10)
index = 0
data = np.loadtxt('code/top-quarks/trained/tmp/data%d' % index)
x_train = data[:, 1:]
y_train = data[:, 0]

clf.fit(x_train, y_train)

x_test = data[:, 1:]
y_test = data[:, 0]

print(clf.score(x_test, y_test))
particles = pd.read_csv("train_sample/event000021100-particles.csv")
particles.head()
