# simple makefile to simplify repetitive build env management tasks under posix

PYTHON := $(shell which python)
CYTHON := $(shell which cython)
NOSETESTS := $(shell which nosetests)

CYTHON_PYX := root_numpy/src/_librootnumpy.pyx
CYTHON_CPP := root_numpy/src/_librootnumpy.cpp
CYTHON_SRC := $(filter-out $(CYTHON_PYX),$(filter-out $(CYTHON_CPP),$(wildcard root_numpy/src/*)))
CYTHON_PYX_SRC := $(filter-out $(CYTHON_PYX),$(wildcard root_numpy/src/*.pyx))

all: $(CYTHON_CPP) clean inplace test

clean-pyc:
	@find . -name "*.pyc" -exec rm {} \;

clean-so:
	@find root_numpy -name "*.so" -exec rm {} \;

clean-build:
	@rm -rf build

clean-html:
	@find root_numpy/src -name "*.html" -exec rm {} \;

clean: clean-build clean-pyc clean-so

$(CYTHON_PYX): $(CYTHON_SRC)

$(CYTHON_CPP): $(CYTHON_PYX)
	@echo "compiling $< ..."
	@$(CYTHON) --cplus --fast-fail --line-directives $<

cython:
	@echo "compiling $(CYTHON_PYX) ..."
	$(CYTHON) --cplus --fast-fail --line-directives $(CYTHON_PYX)

show-cython: clean-html
	@tmp=`mktemp -d`; \
	for pyx in $(CYTHON_PYX_SRC); do \
		echo "compiling $$pyx ..."; \
		name=`basename $$pyx`; \
		cat root_numpy/src/setup.pxi $$pyx > $$tmp/$$name; \
		$(CYTHON) -a --cplus --fast-fail --line-directives -Iroot_numpy/src $$tmp/$$name; \
	done; \
	mv $$tmp/*.html root_numpy/src/; \
	rm -rf $$tmp; \
	for html in root_numpy/src/*.html; do \
		echo "opening $$html ..."; \
		xdg-open $$html; \
	done

in: inplace # just a shortcut
inplace:
	@$(PYTHON) setup.py build_ext -i

install: clean
	@$(PYTHON) setup.py install

install-user: clean
	@$(PYTHON) setup.py install --user

sdist: clean
	@$(PYTHON) setup.py sdist --release

register:
	@$(PYTHON) setup.py register --release

upload: clean
	@$(PYTHON) setup.py sdist upload --release

test-code: inplace
	@$(NOSETESTS) -s -v root_numpy

test-installed:
	@(mkdir -p nose && cd nose && \
	$(NOSETESTS) -s -v --exe root_numpy && \
	cd .. && rm -rf nose)

test-doc:
	@$(NOSETESTS) -s --with-doctest --doctest-tests --doctest-extension=rst \
	--doctest-extension=inc --doctest-fixtures=_fixture docs/

test-coverage: in
	@rm -rf coverage .coverage
	@$(NOSETESTS) -s -v -a '!slow' --with-coverage \
		--cover-erase --cover-branches \
		--cover-html --cover-html-dir=coverage root_numpy
	@xdg-open coverage/index.html

test: test-code test-doc

trailing-spaces:
	@find root_numpy -name "*.py" | xargs perl -pi -e 's/[ \t]*$$//'

doc-clean:
	@make -C docs/ clean

doc: clean doc-clean inplace
	@make -C docs/ html

check-rst:
	@$(PYTHON) setup.py --long-description | rst2html.py > __output.html
	@rm -f __output.html

gh-pages: doc
	@./ghp-import -m "update docs" -r upstream -f -p docs/_build/html/
