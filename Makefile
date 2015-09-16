# simple makefile to simplify repetitive build env management tasks under posix

PYTHON := $(shell which python)
CYTHON := $(shell which cython)
NOSETESTS := $(shell which nosetests)

CYTHON_CORE_PYX := root_numpy/src/_librootnumpy.pyx
CYTHON_TMVA_PYX := root_numpy/tmva/src/_libtmvanumpy.pyx
CYTHON_CORE_CPP := $(CYTHON_CORE_PYX:.pyx=.cpp)
CYTHON_TMVA_CPP := $(CYTHON_TMVA_PYX:.pyx=.cpp)
CYTHON_PYX_SRC := $(filter-out $(CYTHON_CORE_PYX),$(wildcard root_numpy/src/*.pyx))

INTERACTIVE := $(shell ([ -t 0 ] && echo 1) || echo 0)

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	OPEN := open
else
	OPEN := xdg-open
endif

all: clean cython inplace test

clean-pyc:
	@find . -name "*.pyc" -exec rm {} \;

clean-so:
	@find root_numpy -name "*.so" -exec rm {} \;

clean-build:
	@rm -rf build

clean-html:
	@find root_numpy/src -name "*.html" -exec rm {} \;

clean: clean-build clean-pyc clean-so

.SECONDEXPANSION:
%.cpp: %.pyx $$(filter-out $$@,$$(wildcard $$(@D)/*))
	@echo "compiling $< ..."
	$(CYTHON) --cplus --fast-fail --line-directives $<

cython: $(CYTHON_CORE_CPP) $(CYTHON_TMVA_CPP)

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
	if [ "$(INTERACTIVE)" -eq "1" ]; then \
		for html in root_numpy/src/*.html; do \
			echo "opening $$html ..."; \
			$(OPEN) $$html; \
		done; \
	fi;

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
		--cover-min-percentage=100 \
		--cover-html --cover-html-dir=coverage root_numpy
	@if [ "$(INTERACTIVE)" -eq "1" ]; then \
		$(OPEN) coverage/index.html; \
	fi;

test: test-code test-doc

trailing-spaces:
	@find root_numpy -name "*.py" | xargs perl -pi -e 's/[ \t]*$$//'

doc-clean:
	@make -C docs/ clean

doc: doc-clean inplace
	@make -C docs/ html

check-rst:
	@$(PYTHON) setup.py --long-description | rst2html.py > __output.html
	@rm -f __output.html

gh-pages: doc
	@./ghp-import -m "update docs" -r upstream -f -p docs/_build/html/
