default:
	@pandoc \
	-V repository=https://github.com/scikit-hep/root_numpy \
      	-S -o paper.tex -V geometry:margin=1in \
      	--filter pandoc-citeproc paper.md \
	--bibliography paper.bib --biblatex \
      	--template latex.template -H preamble-latex.tex
	@latexmk -pdf paper.tex

clean:
	@latexmk -C
	@find . -name "*.run.xml" -exec rm {} \;
	@find . -name "*.bbl" -exec rm {} \;
