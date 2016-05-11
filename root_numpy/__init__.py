import warnings
import re

# Make sure that DeprecationWarning within this package always gets printed
warnings.filterwarnings('always', category=DeprecationWarning,
                        module='^{0}\.'.format(re.escape(__name__)))

from .setup_utils import root_version_active, get_config

ROOT_VERSION = root_version_active()
config = get_config()

if config is not None:  # pragma: no cover
    root_version_at_install = config.get('ROOT_version', ROOT_VERSION)

    if ROOT_VERSION != root_version_at_install:
        warnings.warn(
            "ROOT {0} is currently active but you "
            "installed root_numpy against ROOT {1}. "
            "Please consider reinstalling root_numpy "
            "for this ROOT version.".format(
                ROOT_VERSION, root_version_at_install),
            RuntimeWarning)

    import numpy
    numpy_version_at_install = config.get('numpy_version', numpy.__version__)

    if numpy.__version__ != numpy_version_at_install:
        warnings.warn(
            "numpy {0} is currently installed but you "
            "installed root_numpy against numpy {1}. "
            "Please consider reinstalling root_numpy "
            "for this numpy version.".format(
                numpy.__version__, numpy_version_at_install),
            RuntimeWarning)

    del root_version_at_install
    del numpy_version_at_install

from ._tree import (
    root2array, root2rec,
    tree2array, tree2rec,
    array2tree, array2root,
    list_trees, list_branches,
    list_directories, list_structures)
from ._hist import fill_hist, fill_profile, hist2array, array2hist
from ._graph import fill_graph
from ._sample import random_sample
from ._array import array
from ._matrix import matrix
from ._evaluate import evaluate
from ._warnings import RootNumpyUnconvertibleWarning
from ._utils import (
    stretch, blockwise_inner_join,
    rec2array, stack, dup_idx)
from .info import __version__


__all__ = [
    'root2array',
    'root2rec',
    'tree2array',
    'tree2rec',
    'array2tree',
    'array2root',
    'hist2array',
    'array2hist',
    'fill_hist',
    'fill_profile',
    'fill_graph',
    'random_sample',
    'array',
    'matrix',
    'evaluate',
    'list_trees',
    'list_branches',
    'list_structures',
    'list_directories',
    'rec2array',
    'stack',
    'stretch',
    'dup_idx',
    'blockwise_inner_join',
    'RootNumpyUnconvertibleWarning',
]
