#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

set -e

sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get -qq update
sudo apt-get -qq install g++-4.8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 90
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 90
sudo apt-get -qq install python-nose python-pip
pip install coverage coveralls

# Install the ROOT binary
time wget --no-check-certificate https://copy.com/rtIyUdxgjt7h/ci/root_builds/root${ROOT}_python${TRAVIS_PYTHON_VERSION}_gcc4.8_x86_64.tar.gz
time tar zxf root${ROOT}_python${TRAVIS_PYTHON_VERSION}_gcc4.8_x86_64.tar.gz
mv root${ROOT}_python${TRAVIS_PYTHON_VERSION}_gcc4.8_x86_64 root
source root/bin/thisroot.sh
