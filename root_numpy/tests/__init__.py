import os
import warnings
import ROOT
import root_numpy as rnp
from numpy.random import RandomState

ROOT.gErrorIgnoreLevel = ROOT.kFatal
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=rnp.RootNumpyUnconvertibleWarning)
RNG = RandomState(42)

from root_numpy.testdata import get_filepath

def load(data):
    if isinstance(data, list):
        return [get_filepath(x) for x in data]
    return get_filepath(data)

import tempfile
from contextlib import contextmanager

@contextmanager
def temp():
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.root')
    tmp_root = ROOT.TFile.Open(tmp_path, 'recreate')
    yield tmp_root
    tmp_root.Close()
    os.close(tmp_fd)
    os.remove(tmp_path)
