import os
import subprocess


def root_flags(root_config='root-config'):
    root_cflags = subprocess.Popen(
        [root_config, '--cflags'],
        stdout=subprocess.PIPE).communicate()[0].strip()
    root_ldflags = subprocess.Popen(
        [root_config, '--libs'],
        stdout=subprocess.PIPE).communicate()[0].strip()
    if sys.version > '3':
        root_cflags = root_cflags.decode('utf-8')
        root_ldflags = root_ldflags.decode('utf-8')
    return root_cflags.split(), root_ldflags.split()


def root_has_feature(feature, root_config='root-config'):
    if os.getenv('NO_ROOT_NUMPY_{0}'.format(feature.upper())):
        # override
        return False
    has_feature = subprocess.Popen(
        [root_config, '--has-{0}'.format(feature)],
        stdout=subprocess.PIPE).communicate()[0].strip()
    if sys.version > '3':
        has_feature = has_feature.decode('utf-8')
    return has_feature == 'yes'


def root_version_installed(root_config='root-config'):
    root_vers = subprocess.Popen(
        [root_config, '--version'],
        stdout=subprocess.PIPE).communicate()[0].strip()
    if sys.version > '3':
        root_vers = root_vers.decode('utf-8')
    return root_vers


def root_version_active():
    import ROOT
    return ROOT.gROOT.GetVersion()


def get_config():
    from pkg_resources import resource_filename
    config_path = resource_filename('root_numpy', 'config.json')
    if not os.path.isfile(config_path):
        return None
    import json
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config
