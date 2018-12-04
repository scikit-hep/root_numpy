default:
	@pandoc \
	-V repository=https://github.com/scikit-hep/root_numpy \
	-V archive_doi=http://doi.org/10.5281/zenodo.592881 \
	-V formatted_doi=10.21105/joss.00307 \
	-S -o paper.tex -V geometry:margin=1in \
      	--filter pandoc-citeproc paper.md \
	--bibliography paper.bib --biblatex \
      	--template latex.template -H preamble-latex.tex
	@latexmk -pdf paper.tex

clean:
	@latexmk -C
	@find . -name "*.run.xml" -exec rm {} \;
	@find . -name "*.bbl" -exec rm {} \;
