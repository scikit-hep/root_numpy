import os
import sys
import warnings
import ROOT
from numpy.random import RandomState
import tempfile
from contextlib import contextmanager
import root_numpy as rnp
from root_numpy.testdata import get_filepath
import threading

LOCK = threading.RLock()

ROOT.gErrorIgnoreLevel = ROOT.kFatal
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=rnp.RootNumpyUnconvertibleWarning)
RNG = RandomState(42)


def load(data):
    if isinstance(data, list):
        return [get_filepath(x) for x in data]
    return get_filepath(data)


@contextmanager
def temp():
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.root')
    tmp_root = ROOT.TFile.Open(tmp_path, 'recreate')
    try:
        yield tmp_root
    finally:
        tmp_root.Close()
        os.close(tmp_fd)
        os.remove(tmp_path)


@contextmanager
def silence_sout():
    LOCK.acquire()
    sys.__stdout__.flush()
    origstdout = sys.__stdout__
    oldstdout_fno = os.dup(sys.__stdout__.fileno())
    devnull = os.open(os.devnull, os.O_WRONLY)
    newstdout = os.dup(1)
    os.dup2(devnull, 1)
    os.close(devnull)
    sys.__stdout__ = os.fdopen(newstdout, 'w')
    try:
        yield
    finally:
        sys.__stdout__ = origstdout
        sys.__stdout__.flush()
        os.dup2(oldstdout_fno, 1)
        LOCK.release()


@contextmanager
def silence_serr():
    LOCK.acquire()
    sys.__stderr__.flush()
    origstderr = sys.__stderr__
    oldstderr_fno = os.dup(sys.__stderr__.fileno())
    devnull = os.open(os.devnull, os.O_WRONLY)
    newstderr = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    sys.__stderr__ = os.fdopen(newstderr, 'w')
    try:
        yield
    finally:
        sys.__stderr__ = origstderr
        sys.__stderr__.flush()
        os.dup2(oldstderr_fno, 2)
        LOCK.release()


@contextmanager
def silence():
    with silence_sout():
        with silence_serr():
            yield
