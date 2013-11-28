import os
from pkg_resources import resource_filename


def get_filepath(name='test.root'):
    return resource_filename('root_numpy', os.path.join('testdata', name))


def get_file(name='test.root'):
    import ROOT
    filepath = get_filepath(name)
    if not os.path.isfile(filepath):
        raise ValueError(
            "root_numpy test data file {0} does not exist".format(filepath))
    return ROOT.TFile(filepath, 'read')
