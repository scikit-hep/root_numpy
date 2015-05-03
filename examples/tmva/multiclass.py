"""
=============================================
Multiclass Classification with NumPy and TMVA
=============================================
"""
from __future__ import print_function
from array import array
import numpy as np
from numpy.random import RandomState
from root_numpy.tmva import add_classification_events, evaluate_reader
from ROOT import TMVA, TFile, TCut

RNG = RandomState(42)

# Construct an example multiclass dataset
n_vars = 5
n_events = 1000
class_0 = RNG.multivariate_normal(
    np.ones(n_vars) * -3, np.diag(np.ones(n_vars)), n_events)
class_1 = RNG.multivariate_normal(
    np.zeros(n_vars), np.diag(np.ones(n_vars)), n_events)
class_2 = RNG.multivariate_normal(
    np.ones(n_vars) * 3, np.diag(np.ones(n_vars)), n_events)
X = np.concatenate([class_0, class_1, class_2])
y = np.ones(X.shape[0])
w = RNG.randint(1, 10, n_events * 3)
y[:class_0.shape[0]] *= 0
y[-class_2.shape[0]:] *= 2
permute = RNG.permutation(y.shape[0])
X = X[permute]
y = y[permute]

# Split into training and test datasets
X_train, y_train, w_train = X[:n_events], y[:n_events], w[:n_events]
X_test, y_test, w_test = X[n_events:], y[n_events:], w[n_events:]

output = TFile('tmva_output.root', 'recreate')
factory = TMVA.Factory('classifier', output,
                       'AnalysisType=Multiclass:'
                       '!V:Silent:!DrawProgressBar')
for n in range(n_vars):
    factory.AddVariable('f{0}'.format(n), 'F')

# Call root_numpy's utility functions to add events from the arrays
add_classification_events(factory, X_train, y_train, weights=w_train)
add_classification_events(factory, X_test, y_test, weights=w_test, test=True)

# Train a BDT
factory.PrepareTrainingAndTestTree(TCut('1'), 'NormMode=EqualNumEvents')
factory.BookMethod('BDT', 'BDTG',
                   'nCuts=20:NTrees=10:MaxDepth=3:'
                   'BoostType=Grad:Shrinkage=0.10')
factory.TrainAllMethods()

# Classify the test dataset with the BDT
reader = TMVA.Reader()
for n in range(n_vars):
    reader.AddVariable('f{0}'.format(n), array('f', [0.]))
reader.BookMVA('BDT', 'weights/classifier_BDTG.weights.xml')
scores = evaluate_reader(reader, 'BDT', X_test)
print("class probabilities:")
print(scores)
print("class probabilties should sum to 1:")
print(np.sum(scores,axis=1))
