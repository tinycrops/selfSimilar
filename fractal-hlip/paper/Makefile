# Makefile for Fractal-HLIP Paper
# Usage: make all, make paper, make appendix, make clean

MAIN_TEX = fractal_hlip_paper.tex
APPENDIX_TEX = appendix.tex
BIB_FILE = references.bib

# Default target
all: paper appendix

# Compile main paper
paper: $(MAIN_TEX) $(BIB_FILE)
	pdflatex $(MAIN_TEX)
	bibtex $(basename $(MAIN_TEX))
	pdflatex $(MAIN_TEX)
	pdflatex $(MAIN_TEX)
	@echo "Main paper compiled: $(basename $(MAIN_TEX)).pdf"

# Compile appendix
appendix: $(APPENDIX_TEX)
	pdflatex $(APPENDIX_TEX)
	pdflatex $(APPENDIX_TEX)
	@echo "Appendix compiled: $(basename $(APPENDIX_TEX)).pdf"

# Clean auxiliary files
clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fdb_latexmk *.fls *.synctex.gz

# Clean everything including PDFs
distclean: clean
	rm -f *.pdf

# Help
help:
	@echo "Available targets:"
	@echo "  all      - Compile both main paper and appendix"
	@echo "  paper    - Compile main paper only"
	@echo "  appendix - Compile appendix only"
	@echo "  clean    - Remove auxiliary files"
	@echo "  distclean- Remove all generated files including PDFs"
	@echo "  help     - Show this help"

.PHONY: all paper appendix clean distclean help 