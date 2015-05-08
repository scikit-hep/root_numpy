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

from nose.tools import assert_raises, assert_true, assert_equal


ROOT.gErrorIgnoreLevel = ROOT.kFatal
RNG = RandomState(42)


class TMVA_Estimator(object):
    def __init__(self, name, n_vars, n_targets=1,
                 method='BDT', task='Classification'):
        self.name = name
        self.n_vars = n_vars
        self.n_targets = n_targets
        self.method = method
        self.task = task
        self.tmpdir = tempfile.mkdtemp()
        self.output = TFile(os.path.join(self.tmpdir, 'tmva_output.root'),
                            'recreate')
        self.factory = TMVA.Factory(name, self.output,
                                    'AnalysisType={0}:Silent'.format(task))
        for n in range(n_vars):
            self.factory.AddVariable('X_{0}'.format(n), 'F')
        if task == 'Regression':
            for n in range(n_targets):
                self.factory.AddTarget('y_{0}'.format(n), 'F')

    def __del__(self):
        self.output.Close()
        shutil.rmtree(self.tmpdir)

    def fit(self, X, y, X_test=None, y_test=None,
            weights=None, weights_test=None,
            signal_label=None, **kwargs):
        # (re)configure settings since deleting a previous Factory resets all
        # this. This is poor design, TMVA.
        config = TMVA.gConfig()
        config.GetIONames().fWeightFileDir = self.tmpdir
        config.SetSilent(True)
        config.SetDrawProgressBar(False)
        self.factory.DeleteAllMethods()

        extra_kwargs = dict()
        if self.task == 'Regression':
            func = rnp.tmva.add_regression_events
        else:
            func = rnp.tmva.add_classification_events
            extra_kwargs['signal_label'] = signal_label

        # test exceptions
        assert_raises(TypeError, func, object(), X, y)
        assert_raises(ValueError, func,
                      self.factory, X, y[:y.shape[0] / 2])
        if weights is not None:
            assert_raises(ValueError, func, self.factory, X, y,
                          weights=weights[:weights.shape[0]/2])
            assert_raises(ValueError, func, self.factory, X, y,
                          weights=weights[:, np.newaxis])

        assert_raises(ValueError, func, self.factory, [[[1, 2]]], [1])
        assert_raises(ValueError, func, self.factory, [[1, 2]], [[[1]]])

        func(self.factory, X, y, weights=weights, **extra_kwargs)
        if X_test is None:
            X_test = X
            y_test = y
            weights_test = weights
        func(self.factory, X_test, y_test,
             weights=weights_test, test=True, **extra_kwargs)

        self.factory.PrepareTrainingAndTestTree(
            TCut('1'), 'NormMode=EqualNumEvents')
        options = []
        for param, value in kwargs.items():
            if value is True:
                options.append(param)
            elif value is False:
                options.append('!{0}'.format(param))
            else:
                options.append('{0}={1}'.format(param, value))
        options = ':'.join(options)
        self.factory.BookMethod(self.method, self.method, options)
        self.factory.TrainAllMethods()

    def predict(self, X, aux=0.):
        reader = TMVA.Reader()
        for n in range(self.n_vars):
            reader.AddVariable('X_{0}'.format(n), array('f', [0.]))
        reader.BookMVA(self.method,
                       os.path.join(self.tmpdir,
                                    '{0}_{1}.weights.xml'.format(
                                        self.name, self.method)))
        assert_raises(TypeError, rnp.tmva.evaluate_reader,
                      object(), self.method, X)
        assert_raises(ValueError, rnp.tmva.evaluate_reader,
                      reader, 'DoesNotExist', X)
        assert_raises(ValueError, rnp.tmva.evaluate_reader,
                      reader, self.method, [[[1]]])
        if self.task != 'Regression':
            assert_raises(ValueError, rnp.tmva.evaluate_reader,
                          reader, self.method, [1, 2, 3])
        output = rnp.tmva.evaluate_reader(reader, self.method, X, aux)
        if ROOT.gROOT.GetVersionInt() >= 60300:
            method = reader.FindMVA(self.method)
            assert_raises(TypeError, rnp.tmva.evaluate_method,
                          object(), X)
            assert_raises(ValueError, rnp.tmva.evaluate_method,
                          method, [[[1]]])
            output_method = rnp.tmva.evaluate_method(method, X, aux)
            assert_array_equal(output, output_method)
        return output


def make_classification(n_features, n_events_per_class, n_classes):
    blobs = []
    for idx in range(n_classes):
        blob = RNG.multivariate_normal(
            np.ones(n_features) * idx * 5,
            np.diag(np.ones(n_features)),
            n_events_per_class)
        blobs.append(blob)
    X = np.concatenate(blobs)
    # class labels
    y = np.repeat(np.arange(n_classes), n_events_per_class) * 2 - 1
    # event weights
    w = RNG.randint(1, 10, n_events_per_class * n_classes)
    # shuffle
    permute = RNG.permutation(y.shape[0])
    X = X[permute]
    y = y[permute]
    return X, y, w


def test_tmva_methodcuts():
    X, y, w = make_classification(2, 300, 2)
    est = TMVA_Estimator('Cuts', 2, method='Cuts')
    est.fit(X, y,
            FitMethod='MC', EffSel=True, SampleSize=100,
            VarProp='FSmart')
    y_predict_1 = est.predict(X, 0.1)
    y_predict_9 = est.predict(X, 0.9)
    assert_true((y_predict_1 != y_predict_9).any())
    assert_true((y_predict_1 <= y_predict_9).all())


def test_tmva_twoclass():
    n_vars = 2
    n_events = 1000
    X, y, w = make_classification(n_vars, n_events, 2)
    X_train, y_train, w_train = X[:n_events], y[:n_events], w[:n_events]
    X_test, y_test, w_test = X[n_events:], y[n_events:], w[n_events:]

    clf = TMVA_Estimator('unweighted', n_vars)
    clf.fit(X_train, y_train, X_test=X_test, y_test=y_test,
            nCuts=20, NTrees=10, MaxDepth=3)
    y_decision = clf.predict(X_test)
    y_predicted = 2 * (y_decision > 0) - 1
    assert_true(np.sum(np.abs(y_predicted - y_test)) < 0.1 * y_test.shape[0])

    clf = TMVA_Estimator('unweighted_label', n_vars)
    clf.fit(X_train, y_train, X_test=X_test, y_test=y_test, signal_label=1,
            nCuts=20, NTrees=10, MaxDepth=3)
    y_decision_label = clf.predict(X_test)
    assert_array_equal(y_decision_label, y_decision)

    # train with weights
    clf = TMVA_Estimator('weighted', n_vars)
    clf.fit(X_train, y_train, X_test=X_test, y_test=y_test,
            weights=w_train, weights_test=w_test,
            nCuts=20, NTrees=10, MaxDepth=3)
    y_decision_weighted = clf.predict(X_test)
    assert_true(np.any(y_decision_weighted != y_decision))

    # unit weights should not change output
    clf = TMVA_Estimator('unit_weights', n_vars)
    clf.fit(X_train, y_train, X_test=X_test, y_test=y_test,
            weights=np.ones(y_train.shape[0]),
            weights_test=np.ones(y_test.shape[0]),
            nCuts=20, NTrees=10, MaxDepth=3)
    y_decision_unit_weights = clf.predict(X_test)
    assert_array_equal(y_decision, y_decision_unit_weights)

    # events can be 1D
    clf = TMVA_Estimator('onedim_events', 1)
    clf.fit(X_train[:, 0], y_train, X_test=X_test[:, 0], y_test=y_test,
            nCuts=20, NTrees=10, MaxDepth=3)


def test_tmva_multiclass():
    n_vars = 2
    n_events = 500
    X, y, w = make_classification(n_vars, n_events, 3)

    # Split into training and test datasets
    X_train, y_train, w_train = X[:n_events], y[:n_events], w[:n_events]
    X_test, y_test, w_test = X[n_events:], y[n_events:], w[n_events:]

    clf = TMVA_Estimator('unweighted', n_vars, task='Multiclass')
    clf.fit(X_train, y_train, X_test=X_test, y_test=y_test,
            nCuts=20, NTrees=10, MaxDepth=3,
            BoostType='Grad', Shrinkage='0.10')
    y_decision = clf.predict(X_test)
    # Class probabilities should sum to one
    assert_array_equal(np.sum(y_decision, axis=1),
                       np.ones(y_decision.shape[0]))


def test_tmva_regression():
    X = np.linspace(0, 6, 100)[:, np.newaxis]
    y = np.sin(X).ravel() + \
        np.sin(6 * X).ravel() + \
        RNG.normal(0, 0.1, X.shape[0])
    w = RNG.randint(1, 10, y.shape[0])

    reg = TMVA_Estimator('regressor', 1, task='Regression')
    reg.fit(np.ravel(X), y, X_test=X, y_test=y,
            nCuts=20, NTrees=10, MaxDepth=3,
            boosttype='AdaBoostR2', SeparationType='RegressionVariance')
    y_predict = reg.predict(np.ravel(X))
    assert_equal(y_predict.ndim, 1)

    # train with weights
    reg = TMVA_Estimator('regressor_weighted', 1, task='Regression')
    reg.fit(X, y, X_test=X, y_test=y, weights=w, weights_test=w,
            nCuts=20, NTrees=10, MaxDepth=3,
            boosttype='AdaBoostR2', SeparationType='RegressionVariance')
    y_predict_weighted = reg.predict(X)
    assert_true(np.any(y_predict_weighted != y_predict))

    # unit weights should not change output
    reg = TMVA_Estimator('regressor_unit_weights', 1, task='Regression')
    reg.fit(X, y, X_test=X, y_test=y,
            weights=np.ones(y.shape[0]), weights_test=np.ones(y.shape[0]),
            nCuts=20, NTrees=10, MaxDepth=3,
            boosttype='AdaBoostR2', SeparationType='RegressionVariance')
    y_predict_unit_weights = reg.predict(X)
    assert_array_equal(y_predict_unit_weights, y_predict)

    # Multi-output
    y_multi = np.c_[y, 1. - y]
    reg = TMVA_Estimator('regressor_multioutput', 1, n_targets=2,
                         method='KNN', task='Regression')
    reg.fit(X, y_multi, X_test=X, y_test=y_multi,
            nkNN=20, ScaleFrac=0.8, SigmaFact=1.0, Kernel='Gaus', UseKernel='F',
            UseWeight='T')
    y_predict = reg.predict(X)
    assert_equal(y_predict.ndim, 2)
