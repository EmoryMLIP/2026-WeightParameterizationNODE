# Build everything: tests, figures, tables, paper.
#
#   make test      run the MATLAB tests and python/test_timeparam.py
#   make figures   compile every figure standalone to PDF
#   make tables    regenerate tables/*.tex from run results
#   make paper     build main.pdf
#   make audit     check every number in the results sections against the ledger
#   make stats     recompute the aggregate statistics the prose quotes
#   make verify    clone into a temp dir and build there, which catches files
#                  that exist locally but are not committed
#   make arxiv     build the submission tarball from a fresh clone
#   make clean     remove build output (keeps results and figures)
#
# Sweeps are not make targets because they take hours. Run one explicitly, for example:
# cd matlab/experiments && ./runNamedSweep.sh runPaperSweep paper 20

# Machine-specific overrides (e.g. PYTHON = ~/venvs/<env>/bin/python) go in
# local.mk, which is gitignored. Without it the defaults below apply.
-include local.mk

MATLAB ?= matlab
PYTHON ?= python3

.PHONY: all test test-matlab test-python figures tables paper audit stats verify arxiv clean

all: figures tables paper

test: test-matlab test-python

test-matlab:
	cd matlab/tests && $(MATLAB) -batch "addpath(genpath('..')); runTests"

test-python:
	cd python && $(PYTHON) test_timeparam.py && $(PYTHON) test_paired_stats.py

figures:
	$(MAKE) -C figures

tables:
	cd matlab/analysis && $(MATLAB) -batch "addpath(genpath('..')); makeTableC1"
	cd python && $(PYTHON) make_tables.py --sweep 2026-10-05-node-final && $(PYTHON) make_degree_efficiency_table.py

paper: figures
	latexmk -pdf -synctex=1 -interaction=nonstopmode main.tex

# Check that every number in the results sections traces to the ledger.
audit:
	cd python && $(PYTHON) audit_numbers.py

# Print every aggregate statistic the prose quotes.
stats:
	cd python && $(PYTHON) paper_statistics.py

# Build from a throwaway clone, so that files excluded by a global gitignore
# or never committed are missing as they would be for a reader.
verify:
	@tmp=$$(mktemp -d) && \
	git clone -q . $$tmp/repo && \
	cd $$tmp/repo && \
	for f in main.tex main.bbl main.bib sn-jnl.cls sn-mathphys-num.bst; do \
	    [ -f $$f ] || { echo "MISSING $$f"; exit 1; }; \
	done && \
	for f in $$(grep -o 'figures/[a-z_0-9]*\.pdf' main.tex | sort -u); do \
	    [ -f $$f ] || { echo "MISSING $$f"; exit 1; }; \
	done && \
	latexmk -pdf -synctex=1 -interaction=nonstopmode -halt-on-error main.tex >/dev/null 2>&1 && \
	pdflatex -synctex=1 -interaction=nonstopmode main.tex > $$tmp/final.log 2>&1 && \
	u=$$(grep -cE 'Reference.*undefined|Citation.*undefined' $$tmp/final.log || true) && \
	if [ "$$u" -ne 0 ]; then echo "FAIL: $$u undefined references on the final pass"; exit 1; fi && \
	echo "verify OK: fresh clone builds, $$(pdfinfo main.pdf | awk '/Pages/{print $$2}') pages, 0 undefined" && \
	rm -rf $$tmp

# arXiv wants the sources, not the repository. Build the tarball from a clone
# so that anything uncommitted is excluded by construction.
arxiv: verify
	@tmp=$$(mktemp -d) && \
	git clone -q . $$tmp/repo && \
	cd $$tmp/repo && \
	tar czf $(CURDIR)/arxiv-submission.tar.gz \
	    main.tex main.bbl sn-jnl.cls sn-mathphys-num.bst tables figures/*.pdf && \
	rm -rf $$tmp && \
	echo "wrote arxiv-submission.tar.gz ($$(du -h $(CURDIR)/arxiv-submission.tar.gz | cut -f1))"

clean:
	latexmk -c main.tex 2>/dev/null || true
	rm -f figures/*.aux figures/*.log
	rm -f arxiv-submission.tar.gz
