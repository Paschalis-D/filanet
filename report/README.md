# Technical report

- `main.tex`: the report, written into the competition's Overleaf template.
- `preamble.tex`, `main.bib`: unchanged from the template.
- `references.bib`: our references. They are cited alongside the template's `main.bib` through `\bibliography{main,references}`.
- `figures/`: the figures. The PDFs are included in the report; the PNGs are previews.
- `main.pdf`: the compiled report.

Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main`, or upload this folder to Overleaf.
