# Literature Survey — AI/ML for Single-Cell Biology & Drug Discovery

**Author**: Rahul M. Sarate  
**Date**: April 2026  
**Purpose**: Supporting evidence and technical foundations for career projects in `bioptimus_project_brainstorm.md` and `high_impact_projects_pharma_ai.md`  
**Target companies**: Bioptimus · Cellarity · Novo Nordisk · Sanofi Digital  

---

## Table of Contents

1. [Single-Cell Foundation Models](#1-single-cell-foundation-models)
2. [Perturbation Prediction & In Silico Screening](#2-perturbation-prediction--in-silico-screening)
3. [Cell State Transitions, EMT & Trajectory Analysis](#3-cell-state-transitions-emt--trajectory-analysis)
4. [Organ Fibrosis & Single-Cell Atlases](#4-organ-fibrosis--single-cell-atlases)
5. [Graph Neural Networks for Gene Regulatory Networks](#5-graph-neural-networks-for-gene-regulatory-networks)
6. [Multi-Omic Integration (scRNA + scATAC)](#6-multi-omic-integration-scrna--scatac)
7. [Deep Generative Models for Single-Cell Data](#7-deep-generative-models-for-single-cell-data)
8. [Federated Learning in Biomedical AI](#8-federated-learning-in-biomedical-ai)
9. [Bulk Deconvolution & Digital Biomarkers](#9-bulk-deconvolution--digital-biomarkers)
10. [Company Science & Strategic Positioning](#10-company-science--strategic-positioning)

---

## 1. Single-Cell Foundation Models

Foundation models pretrained on millions of single-cell transcriptomes are transforming computational biology. They learn gene-gene relationships in a self-supervised manner and can be fine-tuned for diverse downstream tasks with minimal labeled data.

### 1.1 scGPT — Generative Pretrained Transformer for Single-Cell Multi-omics

| Field | Detail |
|---|---|
| **Paper** | Cui, H. et al. "scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI." *Nature Methods* (2024). bioRxiv 2023.04.30.538439 |
| **Lab** | Bo Wang Lab, University of Toronto |
| **Pretraining** | 33 million normal human cells (CellXGene) |
| **Architecture** | Transformer with generative attention masking; gene tokens ranked by expression |
| **Model zoo** | whole-human (recommended), brain (13.2M cells), blood (10.3M), heart (1.8M), lung (2.1M), kidney (814K), pan-cancer (5.7M), continual-pretrained (zero-shot embeddings) |
| **Downstream tasks** | Cell type annotation, perturbation prediction, GRN inference, batch integration, multi-omic integration, reference mapping |
| **Key innovation** | Generative pretraining (unlike BERT-style masked language modeling in Geneformer) enables cell generation and more flexible fine-tuning |
| **Code** | [github.com/bowang-lab/scGPT](https://github.com/bowang-lab/scGPT) — 1.5K stars, MIT license, pip-installable |
| **Zero-shot capabilities** | Reference mapping to 33M+ cells using FAISS (<1s for 10K query cells on GPU); cell embedding for integration without fine-tuning |
| **HuggingFace** | Integration branch available for HF model hub |

**Relevance to Projects:**
- **P3 (Foundation Model for Drug Targets)**: scGPT is the primary model to fine-tune for wound-healing-specific cell type annotation and perturbation prediction
- **N1 (scGPT Fine-tuning)**: Direct implementation target — fine-tune whole-human checkpoint on GSE234269 wound healing data
- Pan-cancer checkpoint enables cross-disease transfer learning

### 1.2 Geneformer — Transfer Learning for Network Biology

| Field | Detail |
|---|---|
| **Paper** | Theodoris, C.V. et al. "Transfer learning enables predictions in network biology." *Nature* 619, 616–624 (2023) |
| **V2 Paper** | Chen, H. et al. "Scaling and quantization of large-scale foundation model enables resource-efficient predictions in network biology." *Nature Computational Science* (2026) |
| **Lab** | Christina Theodoris, Harvard/Broad Institute |
| **V1 Pretraining** | ~30 million human cells (Genecorpus-30M), 10M parameters, input size 2048, ~25K gene vocabulary |
| **V2 Pretraining** | ~104 million human cells (Genecorpus-104M), 104M or 316M parameters, input size 4096, ~20K protein-coding gene vocabulary |
| **Architecture** | BERT-style transformer with rank-value encoding (genes ranked by expression normalized across corpus) |
| **Cancer variant** | Geneformer-V2-104M_CLcancer — continual learning on ~14M cancer cells |
| **Key innovation** | Rank-value encoding deprioritizes housekeeping genes, amplifies cell-state-distinguishing TFs; entirely self-supervised |
| **Downstream tasks** | TF dosage sensitivity, chromatin dynamics, cell type annotation, batch integration, disease classification, in silico perturbation, in silico treatment for therapeutic target discovery |
| **Validated discovery** | Identified novel TF in cardiomyocytes critical for contractile force generation (experimentally validated); predicted therapeutic targets for cardiomyopathy (validated in iPSC models) |
| **Code** | [huggingface.co/ctheodoris/Geneformer](https://huggingface.co/ctheodoris/Geneformer) — 291 likes, Apache 2.0, 5,617 monthly downloads |

**Relevance to Projects:**
- **P1 (Cell State Transition Predictor)**: Geneformer's in silico perturbation capability can predict which gene perturbations shift cell states (e.g., EMT transitions during wound healing)
- **P3 (Foundation Model)**: Benchmark scGPT vs Geneformer for wound healing tasks
- V2's quantization support (8-bit, 4-bit) enables deployment on consumer GPUs — critical for resource-constrained academic labs

### 1.3 Other Foundation Models (Brief Survey)

| Model | Key Feature | Reference |
|---|---|---|
| **scBERT** | BERT-style, gene2vec embeddings, cell type annotation focus | Yang et al. *Nature Machine Intelligence* (2022) |
| **UCE (Universal Cell Embeddings)** | Protein-structure-informed embeddings, species-agnostic | Rosen et al. bioRxiv (2023) |
| **scFoundation** | 100M cells, 20K genes, xTrimoGene architecture, largest single-cell FM | Hao et al. *Nature Methods* (2024) |
| **scPRINT** | Denoising-based, multi-task | Kalfon Lab, built on Geneformer infrastructure |
| **CellPLM** | Cell-centric pretrained language model | bioRxiv (2023) |

**Key Insight**: The field is rapidly converging on transformer-based architectures pretrained on CellXGene/HCA data. Differentiation comes from: (a) encoding strategy (rank-value vs raw expression), (b) pretraining objective (generative vs masked), (c) downstream task specialization.

---

## 2. Perturbation Prediction & In Silico Screening

Predicting cellular responses to genetic/chemical perturbations without performing experiments — critical for drug target discovery and validation.

### 2.1 GEARS — Graph-Enhanced Gene Activation and Repression Simulator

| Field | Detail |
|---|---|
| **Paper** | Roohani, Y., Huang, K. & Leskovec, J. "Predicting transcriptional outcomes of novel multigene perturbations with GEARS." *Nature Biotechnology* (2023) |
| **Lab** | Jure Leskovec, Stanford SNAP Lab |
| **Core idea** | Uses gene-gene knowledge graphs (GO, STRING) + deep learning to predict transcriptional outcomes of single and multi-gene perturbations |
| **Architecture** | GNN on gene interaction graph + perturbation embeddings; predicts post-perturbation gene expression profile |
| **Training data** | Perturb-seq datasets: Norman et al. 2019, Adamson et al. 2016, Dixit et al. 2016 |
| **Key capability** | Predicts outcomes of **combinatorial perturbations** (2-gene knockouts) even when only trained on single-gene data + some combinations |
| **Limitations** | Not designed for cross-cell-type transfer; requires some combinatorial training data; not tested with bulk data |
| **Code** | [github.com/snap-stanford/GEARS](https://github.com/snap-stanford/GEARS) — 354 stars, MIT license, pip-installable (`cell-gears`) |
| **API** | Simple: `gears_model.predict([['CBL', 'CNN1'], ['FEV']])` |

**Relevance to Projects:**
- **N2 (GEARS Perturbation Prediction)**: Direct implementation — train GEARS on Perturb-seq data, predict multi-gene EMT perturbation outcomes
- **P1 (Cell State Transition Predictor)**: GEARS provides the perturbation prediction component; combine with CellRank for trajectory context
- Gene-gene knowledge graph approach directly maps to GRN-based reasoning

### 2.2 CPA — Compositional Perturbation Autoencoder

| Field | Detail |
|---|---|
| **Paper** | Lotfollahi, M. et al. "Predicting cellular responses to complex perturbations in high-throughput screens." *Molecular Systems Biology* (2023) |
| **Lab** | Fabian Theis Lab, Helmholtz Munich |
| **Core idea** | Disentangles perturbation effects from cell state using compositional encoding; enables out-of-distribution prediction |
| **Key capabilities** | Predict unseen drug combinations at various doses; transfer perturbation effects across cell types; estimate dose-response curves; batch effect removal |
| **Architecture** | Variational autoencoder with compositional latent space; separate encoders for cell state, perturbation, and dose |
| **Unique feature** | Supports both CRISPR genetic perturbations AND chemical/drug perturbations with external embeddings (e.g., Morgan fingerprints) |
| **Training** | Built on scvi-tools framework; integrates with Scanpy ecosystem |
| **Code** | [github.com/theislab/cpa](https://github.com/theislab/cpa) — 144 stars, BSD-3 license, pip-installable (`cpa-tools`) |

**Relevance to Projects:**
- **P3 (Foundation Model for Drug Targets)**: CPA bridges genetic perturbations and chemical compounds — essential for translating gene targets to drug candidates
- Dose-response estimation is directly applicable to therapeutic window identification for EMT modulators
- Cross-cell-type transfer enables wound healing predictions informed by other tissue contexts

### 2.3 Replogle et al. 2022 — Genome-Scale Perturb-seq

| Field | Detail |
|---|---|
| **Paper** | Replogle, J.M. et al. "Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq." *Cell* 185, 2559–2575 (2022) |
| **Key data** | Genome-wide CRISPRi screen with single-cell readout in RPE1 and K562 cells; ~2.5M cells, ~10K perturbations |
| **Impact** | Provides the largest training dataset for perturbation prediction models; GEARS v0.1.1 includes preprocessed dataloaders for this data |
| **Relevance** | Gold-standard benchmark for perturbation prediction; essential input for N2 (GEARS) project |

---

## 3. Cell State Transitions, EMT & Trajectory Analysis

Understanding how cells transition between states (especially during wound healing and fibrosis) is central to our research question.

### 3.1 CellRank 2 — Unified Fate Mapping in Multiview Single-Cell Data

| Field | Detail |
|---|---|
| **Paper** | Lange, M. et al. "CellRank for directed single-cell fate mapping." *Nature Methods* 19, 159–170 (2022); CellRank 2: *Nature Methods* (2024) |
| **Lab** | Fabian Theis Lab (scverse ecosystem) |
| **Core idea** | Markov state modeling of cell state transitions; computes fate probabilities and identifies terminal/initial states |
| **V2 advances** | Modular kernel framework: RNA velocity, pseudotime, real-time, metabolic labels, or any custom kernel; unified API |
| **Key capabilities** | Estimate differentiation direction from diverse biological priors; compute macrostates (initial, terminal, intermediate); infer fate probabilities and driver genes; visualize/cluster gene expression trends |
| **Related tools** | moscot (optimal transport), moslin (lineage barcodes), VeloVI (uncertainty-aware velocity), RegVelo (GRN + velocity), CellMapper, CellAnnotator |
| **Code** | [github.com/scverse/cellrank](https://github.com/scverse/cellrank) — 440 stars, BSD-3, part of scverse ecosystem |

**Relevance to Projects:**
- **P1 (Cell State Transition Predictor)**: CellRank is the trajectory inference backbone — identify EMT/MET transitions in wound healing, compute transition probabilities between keratinocyte states
- Driver gene identification pinpoints therapeutic targets controlling cell fate decisions
- CellRank 2's kernel modularity enables combining RNA velocity with experimental time points (control → 3d → 7d → 14d)

### 3.2 EMT in Wound Healing — Key Biology References

| Paper | Key Finding | Relevance |
|---|---|---|
| Nieto, M.A. et al. "EMT: 2016." *Cell* 166, 21–45 (2016) | Comprehensive EMT review: EMT is a spectrum, not binary; partial EMT states are functionally distinct | Partial EMT states during wound healing may be the "fluidity" signatures we measure |
| Dongre, A. & Weinberg, R.A. "New insights into the mechanisms of EMT and implications for cancer." *Nature Reviews Molecular Cell Biology* 20, 69–84 (2019) | EMT plasticity: cells can occupy hybrid E/M states; EMT TFs (Snai1/2, Zeb1/2, Twist1) have non-redundant functions | Direct mapping to our Vim/Cdh1/Snai1/Zeb1 gene signatures |
| Haensel, D. et al. "Defining epidermal basal cell states during skin homeostasis and wound healing." *Cell Reports* 30, 3932–3947 (2020) | Single-cell atlas of mouse skin wound healing: identified distinct basal cell states, wound-activated programs, migration signatures | Directly comparable to our GSE234269 analysis; similar experimental design |
| Joost, S. et al. "Single-cell transcriptomics of traced epidermal and hair follicle stem cells reveals rapid adaptations during wound healing." *Cell Reports* 25, 585–597 (2018) | HFSCs rapidly activate wound healing programs; lineage-traced scRNA-seq reveals cell plasticity during repair | Supports our HFSC analysis; validates tissue fluidity concept |

### 3.3 RNA Velocity & Pseudotime

| Tool | Paper | Key Feature |
|---|---|---|
| **scVelo** | Bergen, V. et al. *Nature Biotechnology* 38, 1408–1414 (2020) | Dynamical model for RNA velocity; estimates latent time; identifies driver genes |
| **Palantir** | Setty, M. et al. *Nature Biotechnology* 37, 451–460 (2019) | Diffusion-based pseudotime; multi-lineage fate probabilities; works well with wound healing time courses |
| **VeloVI** | Gayoso, A. et al. (2024) | Variational inference for velocity; provides uncertainty quantification — critical for noisy wound healing data |

---

## 4. Organ Fibrosis & Single-Cell Atlases

Fibrosis = pathological tissue fluidity. The same EMT/ECM programs that drive wound healing become dysregulated in fibrotic diseases. Direct relevance to Novo Nordisk (NASH/MASH).

### 4.1 Key Single-Cell Fibrosis Atlases

| Study | Organ | GEO | Key Finding | Relevance |
|---|---|---|---|---|
| Ramachandran, P. et al. *Nature* 575, 512–518 (2019) | Liver (cirrhosis) | GSE136103 | Identified scar-associated macrophages (TREM2+CD9+) and pathogenic myofibroblasts; mapped fibrosis niche | NASH/MASH fibrosis for Novo Nordisk; same EMT/ECM genes as wound healing |
| Habermann, A.C. et al. *Science Advances* 6, eaba1972 (2020) | Lung (IPF) | GSE135893 | Aberrant basaloid cells in IPF express KRT17/KRT5/LAMB3; transitional cell states between AT2 and aberrant basaloid | Cross-organ EMT comparison; KRT17 expression parallels wound-edge keratinocytes |
| Muto, Y. et al. *Nature Communications* 12, 2190 (2021) | Kidney (CKD) | GSE151302 | Failed repair proximal tubular cells; identified cell-state-specific fibrosis drivers | Cross-organ fibrosis atlas for P2 (4-organ EMT atlas) |
| Tsoi, L.C. et al. *Journal of Investigative Dermatology* 139, 1480–1489 (2019) | Skin (psoriasis/atopic dermatitis) | GSE121212 | Single-cell atlas of inflammatory skin diseases; keratinocyte state changes | Skin-specific comparison for wound healing |

### 4.2 Cross-Organ Fibrosis Meta-Analysis Approach

**Strategy for P2 (Fibrosis-EMT Atlas):**
1. Harmonize datasets across organs using scVI batch correction
2. Score all cells with shared EMT/ECM/migration gene signatures
3. Identify conserved fibrotic cell states across organs (fibroblast → myofibroblast trajectory is conserved)
4. Map organ-specific vs shared regulatory programs using SCENIC+
5. Build a "fibrosis transition atlas" — a resource for drug target identification across indications

**Key insight**: Hepatic stellate cell → myofibroblast activation in liver mirrors dermal fibroblast → myofibroblast in wound healing. Your wound healing expertise directly translates.

---

## 5. Graph Neural Networks for Gene Regulatory Networks

### 5.1 GNN Architectures for Biology

| Framework | Description | Relevance |
|---|---|---|
| **PyTorch Geometric (PyG)** | Industry-standard GNN library; supports GAT, GCN, GraphSAGE, message passing | Foundation for P5 (Multi-Omic GNN); provides all needed layer types |
| **DGL (Deep Graph Library)** | Alternative to PyG; better for large-scale graphs | Backup framework |
| **GEARS** (see §2.1) | GNN specifically for perturbation prediction on gene interaction graphs | Already covered; shows GNN efficacy for gene networks |

### 5.2 GRN Inference Methods

| Method | Approach | Paper | Relevance |
|---|---|---|---|
| **SCENIC/pySCENIC** | TF-target inference via co-expression + cis-regulatory motif analysis | Aibar et al. *Nature Methods* 14, 1083–1086 (2017) | Gold-standard GRN inference; provides ground truth for GNN training |
| **SCENIC+** | Joint scRNA-seq + scATAC-seq GRN inference via enhancer-gene-TF linkage | Bravo Gonzalez-Blas & De Winter et al. *Nature Methods* (2023) | Multi-omic GRN for P5; links chromatin accessibility to gene regulation |
| **Dictys** | Dynamic GRN inference from multi-omic time series | Wang et al. *Nature Methods* (2023) | Time-resolved GRNs for wound healing dynamics |
| **CellOracle** | Perturbation simulation using GRNs from scATAC-seq; predicts cell identity changes | Kamimoto et al. *Nature* (2023) | Complements GEARS; GRN-guided perturbation prediction |

### 5.3 GNN for GRN — Project Design (P5)

**Architecture concept:**
```
Input: Gene expression matrix + SCENIC+-derived GRN + scATAC peaks
→ Graph construction: Genes as nodes, regulatory edges from SCENIC+
→ Edge features: TF binding scores, chromatin accessibility, distance
→ Node features: Expression, variability, cell-type specificity
→ GAT layers: Multi-head attention captures regulatory hierarchy
→ Output: Predict gene expression changes under perturbation
```

**Key papers on GNNs for biology:**
- Shu, H. et al. "Modeling gene regulatory networks using neural network architectures." *Nature Computational Science* 1, 491–501 (2021)
- Xu, J. et al. "scGPT" (2024) — uses attention weights as proxy for gene-gene regulatory relationships
- Theodoris et al. "Geneformer" (2023) — attention weights encode network hierarchy

---

## 6. Multi-Omic Integration (scRNA + scATAC)

### 6.1 Key Integration Methods

| Method | Paper | Key Feature | Relevance |
|---|---|---|---|
| **SCENIC+** | Bravo Gonzalez-Blas et al. *Nature Methods* (2023) | Enhancer GRN inference from joint scRNA + scATAC | Primary tool for N3 and P5; links chromatin to expression |
| **ArchR** | Granja, J.M. et al. *Nature Genetics* 53, 403–411 (2021) | Comprehensive scATAC-seq analysis: peak calling, gene scores, trajectory, TF motif enrichment | scATAC preprocessing for multi-omic projects |
| **SnapATAC2** | Zhang, K. et al. bioRxiv (2023) | Fast, scalable scATAC analysis in Rust+Python; superior batch correction | Alternative to ArchR for large datasets |
| **Seurat v5 WNN** | Hao, Y. et al. *Cell* 184, 3573–3587 (2021) | Weighted nearest neighbors: integrates any number of modalities | R-based multi-omic integration |
| **MOFA+** | Argelaguet, R. et al. *Genome Biology* 21, 111 (2020) | Multi-Omics Factor Analysis: unsupervised multi-omic decomposition | Identifies shared vs modality-specific factors of variation |
| **SHARE-seq** | Ma, S. et al. *Cell* 183, 1103–1116 (2020) | Joint scRNA + scATAC protocol; demonstrated chromatin potential concept | Experimental protocol for generating paired multi-omic data |
| **MultiVI** | Ashuach, T. et al. *Nature Methods* (2023) | Probabilistic integration of scRNA + scATAC using scvi-tools | Deep learning-based integration; handles missing modalities |

### 6.2 SCENIC+ Deep Dive (Primary Tool for Multi-Omic Projects)

| Feature | Detail |
|---|---|
| **Full name** | Single-Cell regulatory Network Inference and Clustering — Plus |
| **Architecture** | pycisTopic (topic modeling on scATAC) → pycisTarget (motif enrichment) → SCENIC+ (enhancer-gene-TF linkage) |
| **Input** | Paired or separate scRNA-seq + scATAC-seq data |
| **Output** | Enhancer GRN (eGRN): TF → enhancer → target gene triplets; per-cell regulatory activity scores |
| **Perturbation simulation** | Can simulate TF knockouts in silico based on learned eGRN (updated Aug 2024) |
| **Code** | [github.com/aertslab/scenicplus](https://github.com/aertslab/scenicplus) — 251 stars, Python 3.11 |
| **Citation** | Bravo Gonzalez-Blas, C. & De Winter, S. et al. *Nature Methods* (2023) |

**Relevance**: SCENIC+ is the linchpin for P5 (Multi-Omic GNN) — it provides the regulatory edges for the GNN graph. The eGRN can be featurized and used as training signal for graph neural networks predicting regulatory outcomes.

---

## 7. Deep Generative Models for Single-Cell Data

### 7.1 scVI / scvi-tools — The Deep Generative Framework

| Field | Detail |
|---|---|
| **Paper** | Lopez, R. et al. "Deep generative modeling for single-cell transcriptomics." *Nature Methods* 15, 1053–1058 (2018) |
| **Platform paper** | Gayoso, A. et al. "A Python library for probabilistic analysis of single-cell omics data." *Nature Biotechnology* 40, 163–166 (2022) |
| **Lab** | Nir Yosef Lab (Weizmann Institute) — part of scverse ecosystem |
| **Core models** | scVI (RNA), scANVI (semi-supervised annotation), totalVI (CITE-seq), MultiVI (RNA + ATAC), PeakVI (ATAC), Solo (doublet detection), DestVI (spatial deconvolution), scArches (reference mapping) |
| **Architecture** | Variational autoencoder (VAE) with count-based likelihood (negative binomial); explicitly models batch effects and library size |
| **Capabilities** | Dimensionality reduction, batch integration, differential expression, automated annotation, doublet detection, spatial deconvolution |
| **Code** | [github.com/scverse/scvi-tools](https://github.com/scverse/scvi-tools) — 1.6K stars, 446 forks, BSD-3, 82 contributors |
| **Impact** | Part of scverse ecosystem (Nature Biotechnology 2023); de facto standard for probabilistic single-cell analysis |

**Relevance to Projects:**
- **N6 (scVI Deep Generative Model)**: Direct implementation — train scVI on wound healing data for batch correction and latent space learning
- **P2 (Fibrosis Atlas)**: Use scVI/scArches for cross-organ batch integration (harmonize 4 organ datasets)
- scANVI for semi-supervised cell type transfer from reference atlases
- MultiVI for integrating scRNA + scATAC in multi-omic projects

### 7.2 Other Generative Models

| Model | Key Feature | Paper |
|---|---|---|
| **LDVAE** | Linear decoder VAE for interpretable latent space | Svensson et al. (2020) |
| **scGen** | VAE for perturbation response prediction | Lotfollahi et al. *Nature Methods* (2019) |
| **scArches** | Reference mapping using conditional VAE | Lotfollahi et al. *Nature Biotechnology* (2022) |
| **MILO** | Differential abundance testing on KNN graph | Dann et al. *Nature Biotechnology* (2022) |

---

## 8. Federated Learning in Biomedical AI

### 8.1 Flower — The Federated AI Framework

| Field | Detail |
|---|---|
| **Paper** | Beutel, D.J. et al. "Flower: A Friendly Federated Learning Research Framework." arXiv:2007.14390 (2020) |
| **Organization** | Flower Labs (flwrlabs) — spun out of University of Oxford |
| **Core design** | Framework-agnostic (PyTorch, TensorFlow, JAX, scikit-learn, XGBoost, etc.); customizable strategies; simulation + deployment |
| **Scale** | 6.8K GitHub stars, 1.2K forks, 181 contributors, 2500+ dependent projects |
| **Key features** | FedAvg, FedProx, FedNova, FedBN strategies; differential privacy via Opacus; vertical FL; LLM fine-tuning |
| **Healthcare users** | NHS, Owkin, various hospital networks |
| **Baselines** | 30+ federated learning paper reproductions for benchmarking |
| **Code** | [github.com/flwrlabs/flower](https://github.com/flwrlabs/flower) — Apache 2.0 |
| **Training** | DeepLearning.AI course with Andrew Ng available |
| **Latest** | Foundation model support; decentralized training research; Flower Intelligence for on-device LLMs |

### 8.2 Owkin — Federated Learning for Precision Medicine

| Field | Detail |
|---|---|
| **Company** | Owkin (Paris, London, New York, Geneva, Boston) |
| **Core tech** | Federated learning for clinical data across hospital networks; digital pathology |
| **Products** | FedECA (Federated External Control Arm for survival analysis — published as Flower App), MSIntuit (colorectal cancer MSI detection), HE2RNA (predicting gene expression from H&E images) |
| **Sanofi partnership** | Strategic collaboration for AI-driven drug development; Owkin provides federated infrastructure for Sanofi's precision medicine pipeline |
| **Relevance** | Direct connection to P6 (Federated Learning Simulation); Owkin's infrastructure runs on Flower framework |

### 8.3 Key FL Papers in Healthcare/Biology

| Paper | Key Finding | Relevance |
|---|---|---|
| Dayan, I. et al. "Federated learning for predicting clinical outcomes in patients with COVID-19." *Nature Medicine* 27, 1735–1743 (2021) | FL across 20 hospitals (5 continents) for COVID-19 outcome prediction; demonstrated FL matches pooled performance | Proof that FL works for clinical prediction at global scale |
| Pati, S. et al. "Federated learning enables big data for rare cancer boundary detection." *Nature Communications* 13, 7346 (2022) | FL for brain tumor segmentation across 71 institutions | Shows FL for imaging; applicable to spatial transcriptomics |
| Warnat-Herresthal, S. et al. "Swarm learning for decentralized and confidential clinical machine learning." *Nature* 594, 265–270 (2021) | Blockchain-based swarm learning for disease classification | Alternative to centralized FL; relevant for truly decentralized scenarios |

**Strategy for P6 (Federated Learning Simulation):**
1. Simulate multi-site scRNA-seq with batch effects (each site = different GEO dataset)
2. Implement FedAvg with Flower on scVI models — each site trains locally, only shares model gradients
3. Compare: federated vs pooled vs isolated training on cell type classification and DE analysis
4. Add differential privacy (Opacus) and measure privacy-utility tradeoff
5. Benchmark against Owkin's FedECA for survival analysis

---

## 9. Bulk Deconvolution & Digital Biomarkers

### 9.1 CIBERSORTx — Digital Cytometry

| Field | Detail |
|---|---|
| **Paper** | Newman, A.M. et al. "Determining cell type abundance and expression from bulk tissues with digital cytometry." *Nature Biotechnology* 37, 773–782 (2019) |
| **Lab** | Ash Alizadeh Lab + Aaron Newman Lab, Stanford |
| **Core idea** | Deconvolve bulk RNA-seq into cell type proportions using single-cell reference; also impute cell-type-specific gene expression from bulk |
| **Extended tool** | EcoTyper — identifies cell states and ecosystems from bulk/single-cell/spatial data |
| **URL** | [cibersortx.stanford.edu](https://cibersortx.stanford.edu) |

### 9.2 BayesPrism — Bayesian Deconvolution

| Field | Detail |
|---|---|
| **Paper** | Chu, T. et al. "Cell type and gene expression deconvolution with BayesPrism enables Bayesian integrative analysis across bulk and single-cell RNA sequencing in oncology." *Nature Cancer* 3, 505–517 (2022) |
| **Lab** | Charles Danko Lab + Dana Pe'er Lab |
| **Core idea** | Fully Bayesian framework: jointly estimates cell type composition AND cell-type-specific gene expression from bulk RNA-seq; provides uncertainty estimates (CV) |
| **Key advantage** | Posterior distribution provides uncertainty quantification; handles unknown cell types (tumor heterogeneity); S4 object implementation in R |
| **Tumor specialization** | Models tumor heterogeneity via cell state labels; handles unmatched tumor references |
| **Code** | [github.com/Danko-Lab/BayesPrism](https://github.com/Danko-Lab/BayesPrism) |

### 9.3 Digital Biomarker Strategy (P4)

**For Sanofi Digital Biomarker Discovery:**
1. Build single-cell reference from wound healing/fibrosis datasets
2. Apply CIBERSORTx or BayesPrism to deconvolve TCGA/GTEx bulk RNA-seq
3. Correlate deconvolved cell type proportions with clinical outcomes (survival, treatment response)
4. Use `lifelines` for Kaplan-Meier + Cox regression on deconvolved features
5. Validate biomarker signatures across independent cohorts

**Key insight for interviews**: Bulk RNA-seq is the clinical reality (FFPE samples). Deconvolution bridges scRNA-seq research to clinical application — this is exactly what Sanofi and Novo Nordisk need.

---

## 10. Company Science & Strategic Positioning

### 10.1 Cellarity — Cell State-Correcting Therapeutics

| Aspect | Intelligence |
|---|---|
| **Philosophy** | Disease = dysregulated cell states, not single-target failures. Drugs should correct entire cell states |
| **Platform** | Multi-omics mapping (scRNA-seq, scATAC-seq) + AI-driven prediction of chemical interventions to revert cell dysfunction |
| **Foundation models** | Use "single-cell foundation models" to learn disease mechanism language and predict chemical modulators |
| **Lab-in-the-loop** | Iterative cycle: AI prediction → wet lab validation → model refinement |
| **Pipeline** | CLY-124 (Sickle Cell Disease — Phase 1 started June 2025, first-in-class oral globin-switching); Myelofibrosis program; Exploratory hematology; MASH (partnership with Novo Nordisk) |
| **CLY-124 mechanism** | scRNA-seq + AI modeling revealed undiscovered target controlling globin-switching mechanism; increases fetal hemoglobin via oral administration; not genotoxic (unlike hydroxyurea) |
| **Therapeutic areas** | Hematology, immunology (first), then broader disease applications |
| **Novo Nordisk partnership** | MASH program — directly connects Cellarity's cell-state platform to Novo Nordisk's cardiometabolic pipeline |

**Interview positioning**: *"Cellarity's approach — treating the cell state, not a single target — mirrors exactly what I'm building in my wound healing project. I model how tissue fluidity (a complex cell state phenotype) emerges from EMT, ECM remodeling, and mechanotransduction programs. The question isn't 'which gene?' but 'which cell state transition should we modulate?' That's Cellarity's question too."*

### 10.2 Sanofi Digital — AI-Powered R&D

| Aspect | Intelligence |
|---|---|
| **AI strategy (3 pillars)** | Expert AI (deep ML models), Snackable AI (democratized tools), Generative AI (foundation models) |
| **Key tech** | CodonBERT (LLM for mRNA optimization, 10M sequences), target discovery engines (7 novel targets in 1 year), digital biomarkers |
| **Owkin partnership** | Federated learning for precision medicine across hospital networks; FedECA for survival analysis |
| **Open science** | Open Targets consortium; values open-source contributions |
| **Drug focus** | Dupixent (immunology blockbuster), oncology pipeline, mRNA vaccines |

### 10.3 Novo Nordisk Digital Science

| Aspect | Intelligence |
|---|---|
| **Digital hub** | London + Oxford + Cambridge MA |
| **Focus** | Cardiometabolic: obesity (semaglutide), NASH/MASH, diabetes |
| **Cellarity partnership** | MASH program using cell-state-correcting approach |
| **Your angle** | EMT → fibrosis in NASH/MASH. Wound healing EMT = regenerative fibrosis. Understanding both = therapeutic window |

---

## Summary: How Literature Maps to Projects

| Project | Key Literature | Primary Tools |
|---|---|---|
| **P1: Cell State Transition Predictor** | CellRank 2, Geneformer (in silico perturbation), EMT reviews | CellRank, scVelo, Geneformer |
| **P2: Fibrosis-EMT Atlas (4 organs)** | Ramachandran 2019, Habermann 2020, Muto 2021, Haensel 2020 | scVI, SCENIC+, Scanpy |
| **P3: Foundation Model for Drug Targets** | scGPT, Geneformer, CPA | scGPT, CPA, scvi-tools |
| **P4: Digital Biomarker Discovery** | CIBERSORTx, BayesPrism, lifelines | CIBERSORTx, lifelines, Scanpy |
| **P5: Multi-Omic GNN** | SCENIC+, PyTorch Geometric, GEARS | PyG, SCENIC+, ArchR |
| **P6: Federated Learning Simulation** | Flower, Owkin FedECA, Dayan 2021 | Flower, scVI, Opacus |
| **N1: scGPT Fine-tuning** | Cui et al. 2024 | scGPT |
| **N2: GEARS Perturbation** | Roohani et al. 2023, Replogle et al. 2022 | GEARS, PyG |
| **N3: Multi-Omic Integration** | SCENIC+, ArchR, MultiVI | SCENIC+, scvi-tools |
| **N6: scVI Generative Model** | Lopez et al. 2018, Gayoso et al. 2022 | scvi-tools |

---

## Key Databases & Resources

| Resource | URL | What It Provides |
|---|---|---|
| **CellXGene** | cellxgene.cziscience.com | Largest single-cell atlas; source for scGPT pretraining |
| **Human Cell Atlas** | humancellatlas.org | Comprehensive human cell reference |
| **GEO** | ncbi.nlm.nih.gov/geo | Primary datasets: GSE234269, GSE159827, GSE188432 |
| **Tabula Muris** | tabula-muris.ds.czbiohub.org | Mouse single-cell atlas (our organism) |
| **Open Targets** | opentargets.org | Drug target evidence; Sanofi is consortium member |
| **GTEx** | gtexportal.org | Bulk RNA-seq across tissues; for deconvolution validation |
| **TCGA** | cancer.genome.nih.gov | Cancer bulk data; for biomarker discovery |
| **Perturb-seq data** | Via GEARS dataloaders | Genome-scale perturbation screens |
| **STRING** | string-db.org | Protein-protein interaction network; for GNN edge features |

---

## Reading Priority (For Interview Preparation)

### Must-Read (Read the full paper)
1. Cui et al. 2024 — scGPT (*Nature Methods*)
2. Theodoris et al. 2023 — Geneformer (*Nature*)
3. Roohani et al. 2023 — GEARS (*Nature Biotechnology*)
4. Lange et al. 2022 / 2024 — CellRank 1 & 2 (*Nature Methods*)
5. Ramachandran et al. 2019 — Liver fibrosis atlas (*Nature*)

### Should-Read (Abstract + methods + key figures)
6. Lotfollahi et al. 2023 — CPA (*Molecular Systems Biology*)
7. Lopez et al. 2018 — scVI (*Nature Methods*)
8. Bravo Gonzalez-Blas et al. 2023 — SCENIC+ (*Nature Methods*)
9. Newman et al. 2019 — CIBERSORTx (*Nature Biotechnology*)
10. Dayan et al. 2021 — Federated learning in COVID (*Nature Medicine*)

### Good-to-Know (Abstract + intro)
11. Nieto et al. 2016 — EMT review (*Cell*)
12. Replogle et al. 2022 — Genome-scale Perturb-seq (*Cell*)
13. Hao et al. 2021 — Seurat v4 WNN (*Cell*)
14. Gayoso et al. 2022 — scvi-tools platform (*Nature Biotechnology*)
15. Beutel et al. 2020 — Flower FL framework (arXiv)

---

*Generated: April 2026 | Update as new papers are published*  
*Companion to: `bioptimus_project_brainstorm.md` and `high_impact_projects_pharma_ai.md`*
