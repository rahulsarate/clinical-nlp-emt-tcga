# AGENTS.md - Clinical NLP EMT TCGA Project

## Project Context

This workspace is the Clinical NLP / TCGA EMT project split out from the scRNA-seq tissue-fluidity repository on 2026-05-26.

The project links pathology report text features from TCGA cohorts to molecular EMT scores computed from matched bulk RNA-seq.

Primary local folder:

- `D:\VS CODE Project\clinical-nlp-emt-tcga`

GitHub:

- `https://github.com/rahulsarate/clinical-nlp-emt-tcga`
- Visibility: private
- Default branch: `master`

Original sibling scRNA repo:

- Local folder: `D:\VS CODE Project\scRNA_seq`
- GitHub: `https://github.com/rahulsarate/scRNA-seq-for-tissue-fluidity`
- Use that sibling repo for mouse wound-healing scRNA-seq work.

## Repository Boundary

Stay in this repository for tasks involving:

- TCGA pathology reports.
- Clinical NLP and pathology text normalization.
- Weak labels and manual annotation queues.
- Pathology feature extraction.
- Baseline NLP models and biomedical transformer models.
- TCGA bulk RNA EMT scoring.
- Pathology-to-RNA EMT linkage.
- Paths containing `clinical_nlp` or `tcga_pathology_emt`.
- Scripts `10_*.py` through `18_*.py`, plus any clinical NLP validation scripts.

Do not put mouse scRNA-seq wound-healing analysis back into this repo. That belongs in `D:\VS CODE Project\scRNA_seq`.

## Key Files

- `README.md` - project overview and commands.
- `configs/clinical_nlp_config.yaml` - runtime configuration.
- `configs/clinical_nlp_ontology.yaml` - pathology feature ontology.
- `scripts/python/10_download_tcga_pathology_data.py` - GDC metadata and optional file download.
- `scripts/python/11_extract_and_normalize_reports.py` - pathology report extraction and normalization.
- `scripts/python/12_build_pathology_ontology_and_labelset.py` - ontology-derived label artifacts.
- `scripts/python/13_prepare_annotation_queue.py` - weak labels and review queue.
- `scripts/python/14_train_pathology_baselines.py` - regex/context/TF-IDF baselines.
- `scripts/python/15_train_pathology_transformer.py` - biomedical transformer model.
- `scripts/python/16_score_tcga_bulk_emt.py` - bulk RNA EMT scoring.
- `scripts/python/17_link_pathology_to_rna.py` - patient-level linkage matrix.
- `scripts/python/18_update_clinical_nlp_benchmarks.py` - benchmark table updates.
- `scripts/python/utils/clinical_nlp_*` - shared config, I/O, and text helpers.

## Data Safety

Large and sensitive-ish public-data derivatives should remain local unless explicitly approved.

Do not commit:

- TCGA pathology PDFs under `data/metadata/tcga_pathology_emt/report_files/`.
- TCGA RNA count TSV files under `data/counts/tcga_pathology_emt/`.
- Large parquet outputs.
- Model artifacts.
- Logs with large runtime output.

The `.gitignore` is configured to exclude these by default.

## Handoff Notes

The first GitHub snapshot intentionally included scripts, configs, small manifests, summary CSVs, and docs, but excluded large raw/downloaded/generated artifacts.

If future agents need to reproduce the pipeline from scratch, start from `README.md`, validate configs, then run the numbered scripts in order. If local data is missing, rerun the GDC metadata/download step rather than copying large files from the scRNA repo.
