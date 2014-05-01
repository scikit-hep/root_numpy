#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

set -e

sudo apt-get update -qq
sudo apt-get install -qq python-nose python-pip
pip install coverage coveralls

# Install a ROOT binary that we custom-built in a 64-bit Ubuntu VM
# for the correct Python / ROOT version
time wget --no-check-certificate https://copy.com/rtIyUdxgjt7h/ci/root_builds/root_v${ROOT}_python_${TRAVIS_PYTHON_VERSION}.tar.gz
time tar zxf root_v${ROOT}_python_${TRAVIS_PYTHON_VERSION}.tar.gz
mv root_v${ROOT}_python_${TRAVIS_PYTHON_VERSION} root
source root/bin/thisroot.sh
