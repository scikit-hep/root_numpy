from array import array
import numpy as np
from numpy.random import RandomState
from root_numpy.tmva import factory_add_events, reader_evaluate
from ROOT import TMVA, TFile, TCut

RNG = RandomState(42)

# construct an example dataset
n_vars = 5
n_events = 1000
signal = RNG.multivariate_normal(
    np.ones(n_vars), np.diag(np.ones(n_vars)), n_events)
background = RNG.multivariate_normal(
    np.ones(n_vars) * -1, np.diag(np.ones(n_vars)), n_events)
X = np.concatenate([signal, background])
y = np.ones(signal.shape[0] + background.shape[0])
w = RNG.randint(1, 10, n_events * 2)
y[signal.shape[0]:] *= -1
permute = RNG.permutation(y.shape[0])
X = X[permute]
y = y[permute]

# split into training and test datasets
X_train, y_train, w_train = X[:n_events], y[:n_events], w[:n_events]
X_test, y_test, w_test = X[n_events:], y[n_events:], w[n_events:]

output = TFile('tmva_output.root', 'recreate')
factory = TMVA.Factory('classifier', output, 'AnalysisType=Classification')
for n in range(n_vars):
    factory.AddVariable('f{0}'.format(n), 'F')

# call root_numpy's utility functions to add events from the arrays
factory_add_events(factory, X_train, y_train, weights=w_train)
factory_add_events(factory, X_test, y_test, weights=w_test, test=True)

# train a BDT
factory.PrepareTrainingAndTestTree(TCut('1'), 'NormMode=EqualNumEvents')
factory.BookMethod('BDT', 'BDT', 'nCuts=-1:NTrees=10:MaxDepth=3')
factory.TrainAllMethods()

# classify the test dataset with the BDT
reader = TMVA.Reader()
for n in range(n_vars):
    reader.AddVariable('f{0}'.format(n), array('f', [0.]))
reader.BookMVA('BDT', 'weights/classifier_BDT.weights.xml')
scores = reader_evaluate(reader, 'BDT', X_test)
