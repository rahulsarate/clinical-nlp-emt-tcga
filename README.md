# Pathology report analytics linked to molecular EMT

**Python · Clinical NLP · Pathology context · TCGA bulk RNA-seq · Translational research**

A scientific analytics project by **Rahul Sarate, PhD**, investigating how features extracted from TCGA pathology reports can be linked to epithelial–mesenchymal transition (EMT) scores from matched bulk RNA-seq.

The scientific question is whether structured descriptions of tumour pathology align with molecular programmes relevant to cancer progression. The project connects experimental and disease context with text processing, annotation, model development, and patient-level data linkage.

## Relevance to scientific analytics

| Area | Project focus |
| --- | --- |
| Pathology data interpretation | Convert narrative report text into structured features using an explicit ontology |
| Data preparation and QC | Extract and normalise reports, review labels, and check patient/sample linkage |
| Python analytics | Numbered workflows for data acquisition, NLP, EMT scoring, and linkage |
| Scientist collaboration | Annotation queues and expert review of ambiguous pathology language |
| Translational interpretation | Compare pathology-derived features with molecular EMT scores while considering cohort and tissue context |

**Scope:** This repository analyses pathology **report text**, not histology-slide pixels. It should be presented as pathology-informed NLP and molecular analytics, not as an image-segmentation or diagnostic system. The presence of training scripts does not establish validated model performance; consult benchmark artifacts and their provenance before quoting results.

## Project workflow

1. Extract and normalise TCGA pathology report text.
2. Build pathology feature labels and weak-supervision queues.
3. Prepare expert annotation and review.
4. Train baseline and transformer NLP models.
5. Score TCGA bulk RNA EMT signatures.
6. Link pathology-derived features to molecular EMT scores.
7. Record benchmark results and limitations.

## Main entry points

Install the dependencies in `requirements.txt` in an isolated Python environment. Review the configuration and data availability before running the numbered scripts from the repository root:

```bash
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

These are workflow entry points, not a claim that a fresh checkout includes all required data or trained models.

## Key configuration and review paths

- [Runtime configuration](configs/clinical_nlp_config.yaml)
- [Pathology ontology](configs/clinical_nlp_ontology.yaml)
- [Python workflows](scripts/python/)
- [Analysis artifacts](analysis/clinical_nlp/)
- [Project documentation](docs/)

## Evaluation and interpretation

Before reporting scientific conclusions:

- Verify report extraction quality, negation, uncertainty, and missing or ambiguous pathology descriptions.
- Distinguish weak labels from independently reviewed reference labels; report annotation agreement where available.
- Keep reports from the same patient together when creating train/test partitions.
- Verify patient and specimen identifiers, one-to-many matches, and sample exclusions before linking text with RNA data.
- Report class-specific precision/recall and uncertainty on held-out data rather than relying only on aggregate accuracy.
- Account for tumour type, purity/stromal content, batch effects, and multiple comparisons when interpreting pathology–EMT associations.
- Treat associations as hypotheses requiring independent validation, not proof of mechanism or clinical utility.

## Data safety and reproducibility

Large TCGA report files, RNA count files, parquet outputs, and model artifacts should stay local and should not be committed. Preserve source identifiers and document data provenance, cohort definitions, package versions, model settings, and analysis seeds.

This repository is separate from the mouse wound-healing scRNA-seq project. Its focus is pathology report text and matched molecular data.
