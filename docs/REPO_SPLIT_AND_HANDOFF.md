# Repository Split and Handoff

Last updated: 2026-05-26

## What This Repo Is

This repository contains the TCGA pathology report NLP plus molecular EMT linkage project.

Local folder:

- `D:\VS CODE Project\clinical-nlp-emt-tcga`

GitHub:

- `https://github.com/rahulsarate/clinical-nlp-emt-tcga`
- Visibility: private
- Default branch: `master`

## Where It Came From

This project was split out of the mouse scRNA-seq tissue-fluidity repository:

- Local folder: `D:\VS CODE Project\scRNA_seq`
- GitHub: `https://github.com/rahulsarate/scRNA-seq-for-tissue-fluidity`

The split was done because the NLP/TCGA project has a different organism, data modality, workflow, dependency footprint, and GitHub/data-safety profile than the mouse wound-healing scRNA-seq project.

## What Belongs Here

- `configs/clinical_nlp_config.yaml`
- `configs/clinical_nlp_ontology.yaml`
- Clinical NLP scripts in `scripts/python/10_*.py` through `18_*.py`
- `scripts/python/utils/clinical_nlp_*`
- `analysis/clinical_nlp/` summaries and small review/benchmark artifacts
- `data/metadata/tcga_pathology_emt/` manifests and cohorts
- Local-only TCGA pathology report files
- Local-only TCGA RNA count files
- Clinical NLP methods/career planning docs

## What Does Not Belong Here

- Mouse wound-healing scRNA-seq analysis.
- GSE234269, GSE159827, or GSE188432 processing scripts.
- Scanpy/Seurat clustering for wound-healing cells.
- Tissue-fluidity figures for the mouse wound project.
- scRNA dashboard work from `D:\VS CODE Project\scRNA_seq`.

## GitHub Push Policy

Commit and push:

- Source code.
- Config files.
- Ontology files.
- Small manifests and benchmark summaries.
- Documentation.
- Lightweight tests.

Do not commit or push:

- TCGA report PDFs.
- TCGA bulk RNA TSV files.
- Large `.parquet` files.
- Model output directories.
- Large generated logs.

## Practical Rule for Future Agents

If the task says Clinical NLP, TCGA, pathology report, pathology transformer, bulk RNA EMT score, `clinical_nlp`, or `tcga_pathology_emt`, work here.

If the task says scRNA-seq, wound healing, mouse skin, GSE234269, GSE159827, GSE188432, Scanpy clustering, pseudobulk wound DE, or tissue fluidity figures, switch to:

`D:\VS CODE Project\scRNA_seq`
