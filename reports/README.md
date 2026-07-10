# Reports

Draft files prepared from the final university-machine runs:

- `LM_report.tex`: integrated Lab 04 report for LM Part A and Part B.
- `NLU_report.tex`: integrated Lab 05 report for NLU Part A and Part B.
- `REPORT_NOTES.md`: extended notes and tables to help refine the final text.
- `LM_partA_report.md`, `LM_partB_report.md`, `NLU_partA_report.md`, `NLU_partB_report.md`: part-specific notes.

The official course README asks for two mini-reports: one for LM and one for NLU. For convenience, copies of the two LaTeX reports are also placed as:

- `LM/report.tex`
- `NLU/report.tex`

Before submission, replace the TODO name, matricola, and email fields in the `.tex` files.

## Compile

Compile from the folder containing the target `report.tex`:

```bash
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
```

Or from `reports/`:

```bash
pdflatex LM_report.tex
bibtex LM_report
pdflatex LM_report.tex
pdflatex LM_report.tex

pdflatex NLU_report.tex
bibtex NLU_report
pdflatex NLU_report.tex
pdflatex NLU_report.tex
```

If the compiled PDF is too long, trim result rows from the tables first; tables are intended as editing material, not sacred text.
