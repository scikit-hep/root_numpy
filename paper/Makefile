default:
	@pandoc \
	-V repository=https://github.com/scikit-hep/root_numpy \
      	-S -o paper.tex -V geometry:margin=1in \
      	--filter pandoc-citeproc paper.md \
	--bibliography paper.bib --biblatex \
      	--template latex.template
	@latexmk -pdf paper.tex

clean:
	@latexmk -C
	@find . -name "*.run.xml" -exec rm {} \;
	@find . -name "*.bbl" -exec rm {} \;


#-V repository=https://github.com/scikit-hep/root_numpy \
#-V archive_doi=#{archive_doi} \
#-V paper_url=#{paper_url} \
#-V formatted_doi=#{formatted_doi} \
#-V review_issue_url=#{review_issue_url} \
