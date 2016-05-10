#!/bin/bash
# This script is meant to be called by the "script" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

set -e

gcc -dumpversion
g++ -dumpversion
ldd --version
python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"

# Check if ROOT and PyROOT work
python -c "import ROOT; ROOT.TBrowser()"
python -c "from __future__ import print_function; import ROOT; print(ROOT.gROOT.GetVersion())"

export PYTHONPATH=/home/travis/.local/lib/python${TRAVIS_PYTHON_VERSION}/site-packages/:$PYTHONPATH

# Install into the user site-packages directory and run tests on that
time make install
time make test-installed

# Run tests in the local directory with coverage
if [ ! -z ${COVERAGE+x} ] && [ -z ${NOTMVA+x} ]; then
    # COVERAGE is set and TMVA is included in this build
    # so run the coverage
    time make test-coverage </dev/null
fi
