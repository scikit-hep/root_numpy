import os
import re
import subprocess
import numbers
from collections import namedtuple


class ROOTVersion(namedtuple('_ROOTVersionBase',
                             ['major', 'minor', 'micro'])):

    def __new__(cls, *version):
        if len(version) == 1:
            version = version[0]

        if isinstance(version, numbers.Integral):
            if version < 1E4:
                raise ValueError(
                    "{0:d} is not a valid ROOT version integer".format(version))
            return super(ROOTVersion, cls).__new__(
                cls,
                int(version / 1E4),
                int((version / 1E2) % 100),
                int(version % 100))

        if isinstance(version, tuple):
            return super(ROOTVersion, cls).__new__(cls, *version)

        # parse the string version X.YY/ZZ
        match = re.match(
            r"(?P<major>[\d]+)\.(?P<minor>[\d]+)/(?P<micro>[\d]+)", version)
        if not match:
            raise ValueError(
                "'{0}' is not a valid ROOT version string".format(version))
        return super(ROOTVersion, cls).__new__(
            cls,
            int(match.group('major')),
            int(match.group('minor')),
            int(match.group('micro')))


    def __eq__(self, version):
        if not isinstance(version, tuple):
            version = ROOTVersion(version)
        return super(ROOTVersion, self).__eq__(version)

    def __ne__(self, version):
        return not self.__eq__(version)

    def __gt__(self, version):
        if not isinstance(version, tuple):
            version = ROOTVersion(version)
        return super(ROOTVersion, self).__gt__(version)

    def __ge__(self, version):
        if not isinstance(version, tuple):
            version = ROOTVersion(version)
        return super(ROOTVersion, self).__ge__(version)

    def __lt__(self, version):
        if not isinstance(version, tuple):
            version = ROOTVersion(version)
        return super(ROOTVersion, self).__lt__(version)

    def __le__(self, version):
        if not isinstance(version, tuple):
            version = ROOTVersion(version)
        return super(ROOTVersion, self).__le__(version)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return '{0:d}.{1:02d}/{2:02d}'.format(*self)


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
    return ROOTVersion(root_vers)


def root_version_active():
    import ROOT
    return ROOTVersion(ROOT.gROOT.GetVersionInt())


def get_config():
    from pkg_resources import resource_filename
    config_path = resource_filename('root_numpy', 'config.json')
    if not os.path.isfile(config_path):
        return None
    import json
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config
