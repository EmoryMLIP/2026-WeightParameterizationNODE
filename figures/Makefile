# Build every figure standalone to PDF.  Usage: make -C figures
#
# Rebuilds the LaTeX standalones only. node_nfe_elm, node_nfe_cdr and node_nfe_dcr
# are written directly by matplotlib (python/surrogate/build_fig4_averaged.py)
# and have no .tex source, so `make clean` leaves them in place.
TEX  := $(wildcard *.tex)
PDFS := $(TEX:.tex=.pdf)
GENERATED := $(patsubst %.tex,%.pdf,$(TEX))
ORPHANS := $(filter-out $(GENERATED),$(wildcard *.pdf))

all: $(PDFS)
	@for f in $(ORPHANS); do \
	    [ -f "$$f" ] || echo "  note: $$f is written by matplotlib; see python/README.md"; \
	done

%.pdf: %.tex
	pdflatex -interaction=nonstopmode -halt-on-error $< > $*.log 2>&1 || { cat $*.log; exit 1; }
	@rm -f $*.aux $*.log

# Remove only the PDFs built from the .tex sources here, not the matplotlib ones.
clean:
	rm -f *.aux *.log $(GENERATED)

.PHONY: all clean
