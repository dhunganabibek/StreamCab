SHELL := /bin/sh
LATEX := pdflatex
LATEX_FLAGS := -interaction=nonstopmode -halt-on-error

PRESENTATION_TEX := streamcab_presentation.tex
REPORT_TEX := streamcab_report.tex

.PHONY: all presentation report clean

all: presentation report

presentation:
	$(LATEX) $(LATEX_FLAGS) $(PRESENTATION_TEX)
	$(LATEX) $(LATEX_FLAGS) $(PRESENTATION_TEX)

report:
	$(LATEX) $(LATEX_FLAGS) $(REPORT_TEX)
	$(LATEX) $(LATEX_FLAGS) $(REPORT_TEX)

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.fls *.fdb_latexmk
