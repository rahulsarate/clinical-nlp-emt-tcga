# Clinical NLP EMT TCGA

Clinical pathology NLP pipeline for linking TCGA pathology report features to bulk RNA EMT scores.

## Project Scope

- Extract and normalize TCGA pathology report text.
- Build pathology feature labels and weak supervision queues.
- Train baseline and transformer NLP models.
- Score TCGA bulk RNA EMT signatures.
- Link pathology-derived features to molecular EMT scores.

## Main Entry Points

```powershell
python scripts/python/10_download_tcga_pathology_data.py
python scripts/python/11_extract_and_normalize_reports.py
python scripts/python/12_build_pathology_ontology_and_labelset.py
python scripts/python/13_prepare_annotation_queue.py
python scripts/python/14_train_pathology_baselines.py
python scripts/python/15_train_pathology_transformer.py
python scripts/python/16_score_tcga_bulk_emt.py
python scripts/python/17_link_pathology_to_rna.py
python scripts/python/18_update_clinical_nlp_benchmarks.py
```

## Key Configs

- `configs/clinical_nlp_config.yaml`
- `configs/clinical_nlp_ontology.yaml`

## Data Safety

Large TCGA report files, RNA count files, parquet outputs, and model artifacts should stay local and should not be committed.
