import os
import tempfile
import shutil
from array import array
import atexit

import numpy as np
from numpy.testing import assert_array_equal
from numpy.random import RandomState

import ROOT
from ROOT import TFile, TCut, TMVA

import root_numpy as rnp

from nose.tools import assert_raises, assert_true


ROOT.gErrorIgnoreLevel = ROOT.kFatal
RNG = RandomState(42)


class TMVAClassifier(object):
    def __init__(self, name, n_vars):
        self.name = name
        self.n_vars = n_vars
        self.tmpdir = tempfile.mkdtemp()
        self.output = TFile(os.path.join(self.tmpdir, 'tmva_output.root'),
                            'recreate')
        self.factory = TMVA.Factory(name, self.output,
                                    'AnalysisType=Classification:Silent')
        for n in range(self.n_vars):
            self.factory.AddVariable('f{0}'.format(n), 'F')

    def __del__(self):
        self.output.Close()
        shutil.rmtree(self.tmpdir)

    def fit(self, X, y, X_test=None, y_test=None,
            weights=None, weights_test=None,
            signal_label=None):
        # (re)configure settings since deleting a previous Factory resets all
        # this. This is poor design, TMVA.
        config = TMVA.gConfig()
        config.GetIONames().fWeightFileDir = self.tmpdir
        config.SetSilent(True)
        config.SetDrawProgressBar(False)
        self.factory.DeleteAllMethods()

        # test exceptions
        assert_raises(TypeError, rnp.tmva.add_classification_events,
                      object(), X, y)
        assert_raises(ValueError, rnp.tmva.add_classification_events,
                      self.factory, X, y[:y.shape[0] / 2])
        if weights is not None:
            assert_raises(ValueError, rnp.tmva.add_classification_events,
                          self.factory, X, y,
                          weights=weights[:weights.shape[0]/2])
            assert_raises(ValueError, rnp.tmva.add_classification_events,
                          self.factory, X, y,
                          weights=weights[:, np.newaxis])

        rnp.tmva.add_classification_events(
            self.factory, X, y, weights=weights, signal_label=signal_label)
        if X_test is not None and y_test is not None:
            rnp.tmva.add_classification_events(
                self.factory, X_test, y_test,
                weights=weights_test,
                signal_label=signal_label,
                test=True)
        self.factory.PrepareTrainingAndTestTree(
            TCut('1'), 'NormMode=EqualNumEvents')
        self.factory.BookMethod('BDT', 'BDT',
                                'nCuts=20:NTrees=10:MaxDepth=3')
        self.factory.TrainAllMethods()

    def predict(self, X):
        reader = TMVA.Reader()
        for n in range(self.n_vars):
            reader.AddVariable('f{0}'.format(n), array('f', [0.]))
        reader.BookMVA('BDT',
                       os.path.join(self.tmpdir,
                                    '{0}_BDT.weights.xml'.format(self.name)))
        assert_raises(TypeError, rnp.tmva.evaluate_reader,
                      object(), 'BDT', X)
        assert_raises(ValueError, rnp.tmva.evaluate_reader,
                      reader, 'BDT', [1, 2, 3])
        return rnp.tmva.evaluate_reader(reader, 'BDT', X)


def test_tmva():
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
    X_train, y_train, w_train = X[:n_events], y[:n_events], w[:n_events]
    X_test, y_test, w_test = X[n_events:], y[n_events:], w[n_events:]

    clf = TMVAClassifier('unweighted', n_vars)
    clf.fit(X_train, y_train, X_test=X_test, y_test=y_test)
    y_decision = clf.predict(X_test)
    y_predicted = 2 * (y_decision > 0) - 1
    assert_true(np.sum(np.abs(y_predicted - y_test)) < 0.1 * y_test.shape[0])

    clf = TMVAClassifier('unweighted_label', n_vars)
    clf.fit(X_train, y_train, X_test=X_test, y_test=y_test, signal_label=1)
    y_decision_label = clf.predict(X_test)
    assert_array_equal(y_decision_label, y_decision)

    # train with weights
    clf = TMVAClassifier('weighted', n_vars)
    clf.fit(X_train, y_train, X_test=X_test, y_test=y_test,
            weights=w_train, weights_test=w_test)
    y_decision_weighted = clf.predict(X_test)
    assert_true(np.any(y_decision_weighted != y_decision))

    # unit weights should not change output
    clf = TMVAClassifier('unit_weights', n_vars)
    clf.fit(X_train, y_train, X_test=X_test, y_test=y_test,
            weights=np.ones(y_train.shape[0]),
            weights_test=np.ones(y_test.shape[0]))
    y_decision_unit_weights = clf.predict(X_test)
    assert_array_equal(y_decision, y_decision_unit_weights)
