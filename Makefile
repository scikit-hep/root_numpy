# simple makefile to simplify repetitive build env management tasks under posix

PYTHON := $(shell which python)
CYTHON := $(shell which cython)
NOSETESTS ?= nosetests
CTAGS ?= ctags

all: clean inplace test

clean-pyc:
	find . -name "*.pyc" -exec rm {} \;

clean-so:
	find root_numpy -name "*.so" -exec rm {} \;

clean-build:
	rm -rf build

clean-ctags:
	rm -f tags

clean: clean-build clean-pyc clean-so clean-ctags

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

install:
	$(PYTHON) setup.py install

install-user:
	$(PYTHON) setup.py install --user

sdist: clean
	$(PYTHON) setup.py sdist

register:
	$(PYTHON) setup.py register

upload: clean
	$(PYTHON) setup.py sdist upload

test-code: inplace
	$(NOSETESTS) -s -v root_numpy

test-installed:
	(mkdir nose && cd nose && \
	$(NOSETESTS) -s -v --exe root_numpy && \
	cd - && rm -rf nose)

test-doc:
	$(NOSETESTS) -s --with-doctest --doctest-tests --doctest-extension=rst \
	--doctest-extension=inc --doctest-fixtures=_fixture docs/

test-coverage:
	rm -rf coverage .coverage
	$(NOSETESTS) -s --with-coverage --cover-html --cover-html-dir=coverage \
	--cover-package=root_numpy root_numpy

test: test-code test-doc

trailing-spaces:
	find root_numpy -name "*.py" | xargs perl -pi -e 's/[ \t]*$$//'

ctags:
	# make tags for symbol based navigation in emacs and vim
	# Install with: sudo apt-get install exuberant-ctags
	$(CTAGS) -R *

doc: inplace
	make -C docs/ html

cython:
	$(CYTHON) -a --cplus --fast-fail --line-directives root_numpy/src/_librootnumpy.pyx
	$(CYTHON) -a --cplus --fast-fail --line-directives root_numpy/src/_libinnerjoin.pyx

check-rst:
	$(PYTHON) setup.py --long-description | rst2html.py > __output.html
	rm -f __output.html

gh-pages: doc
	./ghp-import -m "update docs" -r upstream -f -p docs/_build/html/
