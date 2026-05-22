# High-Impact Career Projects — Pharma AI & Computational Biology

**Date**: April 17, 2026  
**Focus**: HARD projects only — maximum career impact for interviews  
**Target companies**: Novo Nordisk Digital Science · Sanofi Digital · Cellarity  
**Your background**: Cancer biology, EMT, cancer therapeutics, biomarker studies, immunotherapy, scRNA-seq  

---

## Company Intelligence

### Novo Nordisk Digital Science

| Aspect | Detail |
|---|---|
| **HQ** | Bagsværd, Denmark — R&D Digital Hub in London, research in Oxford, Cambridge MA |
| **Focus** | Cardiometabolic disease (obesity, diabetes, NASH/MASH), rare hematologic disease |
| **AI strategy** | Deep disease understanding + big data; Oxford centre focuses on disruptive cardiometabolic therapies |
| **Tech they use** | Single-cell omics, spatial transcriptomics, digital biomarkers, real-world evidence (RWE), ML for drug discovery |
| **Key modality** | GLP-1 (semaglutide empire) + RNAi (GalXC platform, Lexington MA) |
| **Why EMT matters** | NASH/MASH → liver fibrosis involves EMT of hepatocytes + hepatic stellate cell activation — your EMT expertise maps directly |
| **What they hire for** | Computational biology, data science, bioinformatics, ML engineering — especially cardiometabolic + fibrosis |

**Your angle:** EMT drives fibrosis (the pathological process in NASH/MASH). Your wound healing EMT work is fibrosis in a regenerative context. Flip the framing: *"I study the same EMT programs that drive liver fibrosis — but in wound healing, where these programs are beneficial. Understanding both sides is how you find therapeutic windows."*

---

### Sanofi Digital (Paris)

| Aspect | Detail |
|---|---|
| **HQ** | Paris, France — Global digital transformation |
| **Focus** | Immunology (Dupixent), oncology, rare disease, vaccines (mRNA), neurology |
| **AI strategy** | 3-pillar: Expert AI (deep ML for R&D), Snackable AI (democratized), Generative AI (CodonBERT for mRNA) |
| **Key tech** | Target discovery engines (7 novel targets in 1 year via AI), CodonBERT (LLM for mRNA, 10M sequences), digital biomarkers, RWE |
| **Partnership** | Owkin (federated learning for precision medicine) — directly relevant |
| **What they hire for** | Data science, bioinformatics, computational biology, ML engineering, digital biomarker discovery |
| **Open science** | Open Targets consortium member — values open-source contributions |

**Your angle:** Sanofi's partnership with Owkin = federated learning on clinical data. Your scRNA-seq + ML pipeline shows you can build the computational biology layer that feeds into their precision medicine strategy. The Paris location overlaps with Bioptimus ecosystem.

---

### Cellarity (Cambridge MA)

| Aspect | Detail |
|---|---|
| **HQ** | Cambridge, Massachusetts (Flagship Pioneering portfolio) |
| **Focus** | "Cell as the drug" — cell state-correcting therapeutics (NOT target-based drug discovery) |
| **Core idea** | Disease = dysregulated cell states. Drugs should correct entire cell states, not hit single targets |
| **Platform** | Multi-omics (scRNA-seq core) → single-cell foundation models → predict chemical structures that switch cell states |
| **Pipeline** | CLY-124 (Sickle Cell Disease) — first candidate in clinical studies |
| **Key people** | Sophie Tritschler (Sr Dir, Computational Biology), Mahdi Zamanighomi (Sr Dir, ML), Jim Collins (MIT, co-founder) |
| **What they hire for** | Computational biology (scRNA-seq, multi-omics), ML/AI (foundation models, generative chemistry), perturbation biology |
| **Lab-in-the-loop** | Iterative: predict → synthesize → test → retrain — production ML, not just analysis |

**Your angle:** This is THE company for your skill set. Your scRNA-seq EMT work directly demonstrates cell state analysis. EMT is literally a cell state transition. *"I've been studying cell state transitions before it was a drug discovery paradigm — EMT is the original cell state switch. I can bring biological intuition about state transitions that pure ML engineers lack."*

---

## High-Impact Project Specs

> Every project below is designed to be **directly demonstrable in an interview**, **technically impressive**, and **biologically meaningful** for at least one target company.

---

### P1. Cell State Transition Predictor — EMT as Proof-of-Concept

**Target company:** Cellarity (primary), Novo Nordisk (fibrosis angle), Sanofi (oncology)  
**Difficulty:** ████████░░ (8/10)  
**Time:** 4–5 weeks  

**What:** Build an end-to-end system that: (1) maps cell states from scRNA-seq, (2) identifies transition states (partial EMT), (3) predicts which perturbations push cells between states, (4) validates against known EMT biology. This is a miniature version of Cellarity's core platform.

**Why this is career-defining:** Cellarity's entire thesis is "correct the cell state." You build a working prototype of their approach using your EMT expertise as the biological testbed.

**Data:**
- Your wound healing dataset (GSE234269) — EMT states in fibroblasts/keratinocytes
- Replogle et al. 2022 Perturb-seq (K562 CRISPRi, ~1000 gene KOs) — perturbation reference
- Norman et al. 2019 — combinatorial perturbations for multi-gene predictions
- Cancer EMT reference: GSE131907 (lung cancer) for cross-disease transfer

**Pipeline:**
```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: Cell State Mapping                                │
│  scRNA-seq → scVI latent space → continuous cell states     │
│  Identify: epithelial, partial-EMT, mesenchymal states      │
│  Method: diffusion pseudotime + RNA velocity + palantir     │
├─────────────────────────────────────────────────────────────┤
│  PHASE 2: State Transition Dynamics                         │
│  Compute transition probabilities (CellRank)                │
│  Identify: driver genes at each transition boundary         │
│  Build: state transition graph (nodes=states, edges=paths)  │
├─────────────────────────────────────────────────────────────┤
│  PHASE 3: Perturbation Prediction                           │
│  Train GEARS / CPA on Perturb-seq reference                 │
│  Predict: which KOs reverse mesenchymal → epithelial?       │
│  Rank: therapeutic targets by predicted state-reversal score │
├─────────────────────────────────────────────────────────────┤
│  PHASE 4: Validation & Visualization                        │
│  Validate predictions against literature (Snai1 KO, etc.)   │
│  Interactive dashboard: explore states + perturbation impact │
│  Publication figure: state landscape + perturbation arrows   │
└─────────────────────────────────────────────────────────────┘
```

**Tech stack:** Python, scVI, CellRank, GEARS/CPA, PyTorch, Scanpy, FastAPI + React (dashboard)

**Output:**
- Cell state landscape with transition dynamics (animated UMAP)
- Perturbation response predictions for 50+ EMT-related genes
- Ranked drug target list (which KOs best reverse mesenchymal state)
- Validation: top predictions match published EMT KO experiments
- Interactive dashboard: pick a gene → see predicted cell state shift

**Interview hook:** *"I built a cell state transition predictor — it identifies partial-EMT intermediates, computes transition probabilities, and predicts which gene knockdowns reverse mesenchymal cell states. My top predictions match published CRISPR validation data."*

**Code skeleton:**
```python
import scanpy as sc
import scvi
import cellrank as cr
from gears import PertData, GEARS

# Phase 1: Cell state mapping
adata = sc.read_h5ad("wound_adata.h5ad")
scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="sample_id")
model = scvi.model.SCVI(adata, n_latent=30)
model.train(max_epochs=200)
adata.obsm["X_scVI"] = model.get_latent_representation()

# Phase 2: CellRank transition probabilities
vk = cr.kernels.VelocityKernel(adata).compute_transition_matrix()
ck = cr.kernels.ConnectivityKernel(adata).compute_transition_matrix()
kernel = 0.8 * vk + 0.2 * ck
estimator = cr.estimators.GPCCA(kernel)
estimator.compute_macrostates(n_states=5)  # epithelial, partial-EMT, mesenchymal, etc.
estimator.compute_fate_probabilities()

# Phase 3: GEARS perturbation prediction
pert_data = PertData('./perturb_data')
pert_data.load(data_name='replogle_rpe1')
gears_model = GEARS(pert_data, device='cuda')
gears_model.model_initialize(hidden_size=64)
gears_model.train(epochs=20)

# Predict: what happens to EMT genes after Snai1 KO?
prediction = gears_model.predict(['SNAI1'])
# Compare predicted EMT score shift vs. known biology
```

---

### P2. Fibrosis-EMT Atlas: Wound Healing ↔ Liver ↔ Lung ↔ Kidney

**Target company:** Novo Nordisk (primary — NASH/MASH fibrosis), Sanofi (fibrotic diseases), Cellarity  
**Difficulty:** ███████░░░ (7/10)  
**Time:** 4–5 weeks  

**What:** Build a cross-organ fibrosis atlas comparing EMT/EndoMT programs across 4 tissues (skin wound healing, liver NASH, lung IPF, kidney CKD) using public scRNA-seq data. Identify conserved vs. organ-specific fibrosis gene programs. Score for druggability.

**Why this is career-defining:** Novo Nordisk's NASH pipeline needs computational biologists who understand fibrosis at single-cell resolution. By comparing your wound healing EMT data to liver fibrosis, you demonstrate the exact translational thinking they need.

**Biological insight:** Fibrosis is pathological wound healing. The same EMT programs (Snai1, Col1a1, Fn1, Tgfb1) drive both — but in fibrosis they don't resolve. Your project makes this comparison quantitative.

**Data:**
| Dataset | Organ | Disease | Cells | GEO |
|---|---|---|---|---|
| GSE234269 | Skin | Wound healing | ~50k | Your current project |
| GSE136103 | Liver | NASH/Cirrhosis | ~100k | Ramachandran et al. 2019, *Nature* |
| GSE135893 | Lung | IPF (pulmonary fibrosis) | ~115k | Habermann et al. 2020, *Sci Adv* |
| GSE151302 | Kidney | CKD (renal fibrosis) | ~60k | Muto et al. 2021, *PNAS* |

**Pipeline:**
```
4 organ-level scRNA-seq datasets →
Harmonize: scVI/Harmony batch correction across organs →
Cross-organ cell type mapping (fibroblast subtypes, myofibroblasts, macrophages) →
EMT / fibrosis gene signature scoring per cell →
Identify: conserved fibrosis core program (shared across all organs) →
Identify: organ-specific programs (liver-only, lung-only, etc.) →

Analysis:
├── Conserved fibrosis TFs (SCENIC cross-organ comparison)
├── Temporal dynamics: wound healing resolves, NASH doesn't — WHY?
├── Drug target scoring: which conserved genes are druggable? (DGIdb lookup)
└── Publication figure: cross-organ fibrosis atlas UMAP + heatmap
```

**Tech stack:** Python, Scanpy, scVI, pySCENIC, DGIdb API, matplotlib, ComplexHeatmap (R)

**Output:**
- Cross-organ integrated UMAP (4 tissues, harmonized)
- Conserved fibrosis gene signature (top 50 genes shared across all 4 organs)
- Organ-specific programs table (CSV + heatmap)
- Druggable target ranking (DGIdb cross-reference)
- Key insight: "Resolution genes" active in wound healing but silenced in NASH → therapeutic targets
- Publication-quality figure panel

**Interview hook:**
- **Novo Nordisk:** *"I identified 50 conserved fibrosis genes across 4 organs — and 12 of them are druggable targets not yet explored in NASH. My wound healing data revealed 'resolution genes' that shut off fibrosis — these are absent in your NASH datasets."*
- **Sanofi:** *"I built a cross-disease fibrosis atlas that could feed into your target discovery engines — the conserved programs suggest common mechanisms you could exploit across indications."*

```python
# Cross-organ integration with scVI
import scvi
import scanpy as sc

datasets = {
    'skin_wound': sc.read_h5ad('GSE234269_processed.h5ad'),
    'liver_nash': sc.read_h5ad('GSE136103_processed.h5ad'),
    'lung_ipf': sc.read_h5ad('GSE135893_processed.h5ad'),
    'kidney_ckd': sc.read_h5ad('GSE151302_processed.h5ad'),
}

# Concatenate with organ label
adata = sc.concat(datasets, label='organ', keys=datasets.keys())

# scVI integration — batch = organ + dataset
scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="organ")
model = scvi.model.SCVI(adata, n_latent=30)
model.train(max_epochs=300)

adata.obsm["X_scVI"] = model.get_latent_representation()
sc.pp.neighbors(adata, use_rep="X_scVI")
sc.tl.umap(adata)

# Score fibrosis signatures
fibrosis_core = ['Col1a1', 'Col3a1', 'Fn1', 'Vim', 'Acta2', 'Tgfb1', 'Lox', 'Loxl2']
sc.tl.score_genes(adata, gene_list=fibrosis_core, score_name='fibrosis_score')
```

---

### P3. Single-Cell Foundation Model for Drug Target Discovery

**Target company:** Cellarity (primary), Sanofi (target discovery engines)  
**Difficulty:** █████████░ (9/10)  
**Time:** 5–6 weeks  

**What:** Fine-tune a single-cell foundation model (scGPT or Geneformer) to: (1) classify disease cell states, (2) predict gene perturbation effects, (3) rank drug targets by predicted cell-state correction score. Validate on EMT biology, then apply to NASH fibrosis.

**Why this is career-defining:** Cellarity uses "single-cell foundation models" (their exact words from platform page). Sanofi's target discovery engines use deep neural networks. You build a working version with biological validation — not just ML metrics.

**Data:**
- Pre-training: scGPT checkpoint (33M cells, HuggingFace)
- Fine-tuning: Your wound healing data + GSE136103 (liver NASH) + GSE131907 (lung cancer)
- Evaluation: held-out cell type labels + known EMT biology + DGIdb druggability

**Pipeline:**
```
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: Foundation Model Setup                            │
│  Load pre-trained scGPT / Geneformer from HuggingFace       │
│  Tokenize your scRNA-seq data (gene expression → tokens)    │
├─────────────────────────────────────────────────────────────┤
│  STAGE 2: Fine-Tune on Disease Cell States                  │
│  Task A: Cell type classification (10 types in wound data)  │
│  Task B: EMT state prediction (epi / partial / mes)         │
│  Task C: Disease condition prediction (control vs wound_7d) │
│  Compare: fine-tuned vs. zero-shot vs. Scanpy baseline      │
├─────────────────────────────────────────────────────────────┤
│  STAGE 3: Gene Program Discovery via Attention              │
│  Extract attention weights → identify gene-gene interactions│
│  Cluster attention patterns → discover EMT gene programs    │
│  Compare: attention-derived programs vs. SCENIC regulons    │
├─────────────────────────────────────────────────────────────┤
│  STAGE 4: In-Silico Perturbation Screen                     │
│  Mask each gene → predict expression of remaining genes     │
│  Score: which gene masks most shift cells toward epithelial? │
│  Rank: drug target candidates by state-correction potential  │
│  Cross-reference: DGIdb for druggability                    │
├─────────────────────────────────────────────────────────────┤
│  STAGE 5: Transfer to NASH Fibrosis                         │
│  Apply fine-tuned model to liver NASH scRNA-seq              │
│  Predict: which genes reverse hepatic stellate cell          │
│  activation (fibrosis driver)?                              │
│  Deliverable: ranked target list for Novo Nordisk angle     │
└─────────────────────────────────────────────────────────────┘
```

**Tech stack:** Python, PyTorch, HuggingFace Transformers, Flash Attention, scGPT/Geneformer, Scanpy, DGIdb API

**Output:**
- Fine-tuned scGPT with EMT specialization (model weights on HuggingFace)
- Benchmark table: scGPT vs. Geneformer vs. Scanpy on cell type / EMT classification
- Attention-derived gene programs (novel EMT interactions)
- In-silico perturbation screen: top 20 drug targets for EMT reversal
- Transfer to NASH: top 10 targets for hepatic fibrosis resolution
- Interactive Gradio demo: input a gene → see predicted cell state shift

**Interview hook:**
- **Cellarity:** *"I fine-tuned scGPT on EMT cell states and used in-silico perturbation screening to rank drug targets — the model correctly prioritized Snai1 and Twist1, and predicted 3 novel targets I validated against literature."*
- **Sanofi:** *"My foundation model approach to target discovery mirrors your AI target engines — I delivered 7 novel target candidates from a single scRNA-seq dataset."*

```python
# scGPT fine-tuning skeleton
from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
import torch

# Load pre-trained
vocab = GeneVocab.from_file("scgpt_vocab.json")
model = TransformerModel(
    ntoken=len(vocab), d_model=512, nhead=8, 
    nlayers=12, nlayers_cls=3, n_cls=3,  # 3 EMT states
)
model.load_pretrained("scGPT_human")

# Fine-tune on EMT classification
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
for epoch in range(10):
    for batch in dataloader:
        logits = model(batch['gene_ids'], batch['expression_values'])
        loss = F.cross_entropy(logits, batch['emt_state_labels'])
        loss.backward()
        optimizer.step()

# In-silico perturbation: mask Snai1 → predict expression changes
masked_input = mask_gene(batch, gene='SNAI1')
predicted = model(masked_input)
emt_shift = compute_emt_score(predicted) - compute_emt_score(original)
```

---

### P4. AI-Powered Digital Biomarker Discovery Platform

**Target company:** Sanofi (primary — digital biomarkers + RWE), Novo Nordisk (Oxford RWE centre)  
**Difficulty:** ████████░░ (8/10)  
**Time:** 4–5 weeks  

**What:** Build an end-to-end platform that: (1) takes a disease + scRNA-seq dataset as input, (2) automatically identifies cell-type-specific biomarker signatures, (3) deconvolves bulk TCGA/clinical RNA-seq to estimate these signatures in patients, (4) correlates with clinical outcomes (survival, treatment response), (5) outputs a ranked biomarker panel with clinical evidence scores.

**Why this is career-defining:** Sanofi explicitly uses AI to "discover and validate digital biomarkers." Novo Nordisk's Oxford centre combines "deep disease understanding with big data." Your platform does both — it goes from single-cell biology to clinical validation in one automated pipeline.

**Data:**
- scRNA-seq reference: Your wound healing data + cancer datasets (lung, breast, CRC)
- Clinical validation: TCGA (bulk RNA + survival + treatment data, 33 cancer types)
- Drug response: GDSC (Genomics of Drug Sensitivity in Cancer) — compound response data
- Real-world: GEO clinical trial datasets (GSE123813 anti-PD1 melanoma)

**Pipeline:**
```
┌─────────────────────────────────────────────────────────────┐
│  MODULE 1: Single-Cell Signature Extraction                 │
│  scRNA-seq → cell type annotation → per-cell-type DE        │
│  Output: signature gene sets per cell type per condition     │
├─────────────────────────────────────────────────────────────┤
│  MODULE 2: Bulk RNA Deconvolution                           │
│  TCGA bulk RNA-seq → BayesPrism / CIBERSORTx deconvolution │
│  Input: scRNA-seq reference signatures                      │
│  Output: per-patient cell type fractions + activity scores  │
├─────────────────────────────────────────────────────────────┤
│  MODULE 3: Clinical Outcome Association                     │
│  Survival analysis: Cox PH for each signature score         │
│  Treatment response: logistic regression for ICB responders │
│  SHAP: which cell types / genes drive predictions?          │
├─────────────────────────────────────────────────────────────┤
│  MODULE 4: Biomarker Ranking & Report                       │
│  Score = statistical significance × effect size × druggability│
│  Output: ranked biomarker panel (CSV + interactive report)  │
│  Auto-generate: Kaplan-Meier curves, forest plots, SHAP     │
├─────────────────────────────────────────────────────────────┤
│  MODULE 5: AI Agent Orchestration                           │
│  LLM agent interprets results in biological context         │
│  Generates: 1-page biomarker brief per candidate            │
│  Format: Quarto report + interactive dashboard              │
└─────────────────────────────────────────────────────────────┘
```

**Tech stack:** Python, Scanpy, BayesPrism (R via rpy2), lifelines (survival), XGBoost, SHAP, FastAPI + React, LangChain (agent)

**Output:**
- Automated biomarker discovery pipeline (CLI + API)
- Validation on known biomarkers: CD8 T-cell infiltration predicts anti-PD1 response (sanity check)
- Novel biomarker candidates with clinical evidence scores
- Interactive dashboard: explore biomarkers, survival curves, deconvolution
- AI-generated biomarker briefs (1-page summaries per candidate)

**Interview hook:**
- **Sanofi:** *"I built an automated digital biomarker platform that goes from scRNA-seq to clinical validation — it rediscovered CD8 T-cell infiltration as an anti-PD1 biomarker and identified 5 novel candidates with survival associations."*
- **Novo Nordisk:** *"My platform integrates single-cell signatures with real-world clinical data — exactly the kind of translational pipeline your Oxford centre is building for cardiometabolic disease."*

```python
# Module 3: Survival analysis
from lifelines import CoxPHFitter
import pandas as pd

# patient_features: columns = [cell_type_fractions, signature_scores, clinical_vars]
# patient_outcomes: columns = [time_to_event, event_observed]
df = pd.concat([patient_features, patient_outcomes], axis=1)

cph = CoxPHFitter()
cph.fit(df, duration_col='time_to_event', event_col='event_observed')
cph.print_summary()
# HR for fibroblast_emt_score: 1.8 (p=0.002) → EMT predicts worse survival

# SHAP for interpretability
import shap
from xgboost import XGBClassifier

model = XGBClassifier(random_state=42)
model.fit(X_train, y_train)  # y = responder vs non-responder
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)  # which features drive response prediction?
```

---

### P5. Multi-Omic Regulatory Network Atlas with GNN

**Target company:** Cellarity (primary — multi-omics platform), Sanofi (target discovery)  
**Difficulty:** █████████░ (9/10)  
**Time:** 5–6 weeks  

**What:** Integrate scRNA-seq + scATAC-seq (chromatin accessibility) to build a gene regulatory network, then train a Graph Neural Network to predict: (a) which TF→target edges are active in each cell state, (b) which TFs are master regulators of EMT, (c) how the network rewires during wound healing. This maps directly to Cellarity's "map networks of cell and gene interactions" platform description.

**Why this is career-defining:** Cellarity's platform page says they "map complex intercellular interactions" using "multi-omics." A GNN on regulatory networks built from multi-omic data is exactly their computational approach.

**Data:**
- 10x Multiome public data: PBMC (validation) + SHARE-seq skin (Ma et al. 2020, directly relevant)
- Your wound healing scRNA-seq (GSE234269) — expression layer
- JASPAR TF motif database — regulatory prior knowledge
- STRING / SCENIC — protein interaction + regulon data

**Pipeline:**
```
scRNA-seq + scATAC-seq (SHARE-seq skin data) →

LAYER 1: Chromatin Accessibility
├── ArchR / SnapATAC2 → peak calling → motif enrichment
├── chromVAR → TF activity scores per cell
└── Peak-to-gene linkage (correlation-based)

LAYER 2: Gene Regulatory Network
├── pySCENIC → TF regulons from expression
├── Integrate: ATAC motif support + expression correlation
└── Build: directed graph (TF → target, weighted by evidence)

LAYER 3: Graph Neural Network
├── Nodes: genes (features = expression + chromatin + motif scores)
├── Edges: regulatory links (features = correlation + motif + distance)
├── Architecture: GAT (Graph Attention Network) with multi-head attention
├── Task 1: Predict EMT master regulators (node classification)
├── Task 2: Predict active edges per wound phase (edge classification)
├── Task 3: Predict cell state from local GRN topology (graph classification)
└── Interpretability: attention weights → regulatory logic

LAYER 4: Validation & Novelty
├── Known EMT TFs (Snai1, Twist1, Zeb1) should rank top
├── Novel predictions → literature validation
├── Therapeutic angle: which TFs have druggable upstream regulators?
└── Dynamic visualization: GRN rewiring animation across wound phases
```

**Tech stack:** Python, PyTorch Geometric (GATConv), ArchR/SnapATAC2, pySCENIC, JASPAR, NetworkX, D3.js (visualization)

**Output:**
- Multi-omic GRN (>5000 nodes, >50000 edges) with evidence weights
- Trained GAT model with interpretable attention weights
- EMT master regulator ranking (validated against known biology)
- GRN rewiring dynamics (control → wound_3d → wound_7d → wound_14d)
- Novel TF predictions with chromatin evidence
- Interactive network visualization (D3.js force-directed graph)
- Publication-quality figure: GRN heatmap + attention weights

**Interview hook:**
- **Cellarity:** *"I built a multi-omic gene regulatory network from paired scRNA+scATAC data and trained a GNN to identify master regulators of EMT. The model correctly ranked known EMT TFs and predicted novel regulators with chromatin evidence — this is exactly how your platform maps cellular networks."*
- **Sanofi:** *"My GRN-based target discovery pipeline identified druggable upstream regulators of fibrosis TFs — targets your AI engines could evaluate for therapeutic potential."*

```python
# GAT for Gene Regulatory Networks
import torch
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data

class GRN_GAT(torch.nn.Module):
    """Graph Attention Network for gene regulatory network analysis."""
    def __init__(self, in_features: int, hidden: int, out_classes: int, heads: int = 4):
        super().__init__()
        self.conv1 = GATConv(in_features, hidden, heads=heads, dropout=0.3)
        self.conv2 = GATConv(hidden * heads, hidden, heads=heads, dropout=0.3)
        self.conv3 = GATConv(hidden * heads, out_classes, heads=1, concat=False)

    def forward(self, x, edge_index, return_attention: bool = False):
        x, attn1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = torch.relu(x)
        x, attn2 = self.conv2(x, edge_index, return_attention_weights=True)
        x = torch.relu(x)
        x, attn3 = self.conv3(x, edge_index, return_attention_weights=True)
        if return_attention:
            return x, (attn1, attn2, attn3)
        return x

# Build graph from SCENIC + ATAC data
# node_features: [expression, chromatin_accessibility, motif_score, degree_centrality]
# edge_features: [tf_target_correlation, motif_support, genomic_distance]
graph = Data(
    x=node_features,        # (n_genes, n_features)
    edge_index=edge_index,  # (2, n_edges)
    y=emt_role_labels,       # 0=passenger, 1=driver, 2=repressor
)

model = GRN_GAT(in_features=node_features.shape[1], hidden=64, out_classes=3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    out = model(graph.x, graph.edge_index)
    loss = torch.nn.functional.cross_entropy(out, graph.y)
    loss.backward()
    optimizer.step()
```

---

### P6. Federated Learning Simulation for Multi-Site Biomarker Discovery

**Target company:** Sanofi (primary — Owkin partnership), Novo Nordisk (multi-site clinical trials)  
**Difficulty:** ████████░░ (8/10)  
**Time:** 3–4 weeks  

**What:** Simulate federated learning across 4 "hospital sites" (each holding a different scRNA-seq dataset). Train a shared biomarker discovery model without exchanging raw data. Show that federated models match centralized performance for EMT/fibrosis biomarker identification.

**Why this is career-defining:** Sanofi's partnership with Owkin is built on federated learning. Novo Nordisk runs multi-site clinical trials where data can't leave hospitals. Demonstrating you understand privacy-preserving ML for biology is rare and highly valued.

**Data (simulated sites):**
| Site | Dataset | Organ | Cells |
|---|---|---|---|
| Site A | GSE234269 | Skin (wound) | ~50k |
| Site B | GSE136103 | Liver (NASH) | ~100k |
| Site C | GSE135893 | Lung (IPF) | ~115k |
| Site D | GSE131907 | Lung (cancer) | ~200k |

**Pipeline:**
```
┌────────────────────────────────────────────────────────────────┐
│  SIMULATION FRAMEWORK                                          │
│                                                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ Site A   │  │ Site B   │  │ Site C   │  │ Site D   │      │
│  │ (Skin)   │  │ (Liver)  │  │ (Lung)   │  │ (Cancer) │      │
│  │ Local    │  │ Local    │  │ Local    │  │ Local    │      │
│  │ scVI     │  │ scVI     │  │ scVI     │  │ scVI     │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘      │
│       │              │              │              │            │
│       └──────────────┴──────────────┴──────────────┘            │
│                           │                                     │
│              ┌────────────▼────────────┐                        │
│              │  AGGREGATION SERVER     │                        │
│              │  FedAvg / FedProx       │                        │
│              │  Aggregate model weights│                        │
│              │  (NO raw data shared)   │                        │
│              └────────────┬────────────┘                        │
│                           │                                     │
│              ┌────────────▼────────────┐                        │
│              │  GLOBAL MODEL           │                        │
│              │  Shared biomarker       │                        │
│              │  signatures             │                        │
│              └─────────────────────────┘                        │
└────────────────────────────────────────────────────────────────┘

Evaluation:
├── Federated model vs. centralized model (same data combined)
├── Per-site performance vs. global performance
├── Biomarker reproducibility across sites
└── Privacy analysis: differential privacy budget (ε)
```

**Tech stack:** Python, Flower (FL framework), scVI, PyTorch, Scanpy

**Output:**
- Federated learning simulation with 4 sites
- Performance comparison: federated vs. centralized (AUC, F1, Pearson correlation)
- Shared biomarker signatures discovered without data exchange
- Privacy guarantees (differential privacy metrics)
- Paper-ready comparison table + visualization

**Interview hook:**
- **Sanofi:** *"I simulated the Owkin federated learning approach — 4 hospital sites each with scRNA-seq, no data exchange, yet the federated model identified the same fibrosis biomarkers as centralized analysis. This is directly how your Owkin partnership operates."*
- **Novo Nordisk:** *"For your multi-site clinical trials, I've demonstrated that privacy-preserving federated learning discovers the same biomarkers as pooled analysis — meaning you can discover targets across hospital sites without GDPR violations."*

```python
# Flower federated learning simulation
import flwr as fl
from flwr.client import NumPyClient
import scvi
import scanpy as sc

class ScRNAClient(NumPyClient):
    """Federated learning client for scRNA-seq biomarker discovery."""
    def __init__(self, site_data_path: str):
        self.adata = sc.read_h5ad(site_data_path)
        scvi.model.SCVI.setup_anndata(self.adata, layer="counts")
        self.model = scvi.model.SCVI(self.adata, n_latent=20)

    def get_parameters(self, config):
        return [p.cpu().numpy() for p in self.model.module.parameters()]

    def fit(self, parameters, config):
        # Set global parameters, train locally
        self._set_parameters(parameters)
        self.model.train(max_epochs=10)
        return self.get_parameters(config), len(self.adata), {}

    def evaluate(self, parameters, config):
        self._set_parameters(parameters)
        elbo = self.model.get_elbo()
        return float(-elbo), len(self.adata), {"elbo": float(elbo)}

# Start federated simulation
fl.simulation.start_simulation(
    client_fn=lambda cid: ScRNAClient(f"site_{cid}_data.h5ad"),
    num_clients=4,
    config=fl.server.ServerConfig(num_rounds=20),
    strategy=fl.server.strategy.FedAvg(),
)
```

---

## Project Impact Matrix

| Project | Cellarity | Novo Nordisk | Sanofi | Key Differentiator |
|---|---|---|---|---|
| **P1** Cell State Predictor | ★★★★★ | ★★★☆☆ | ★★★☆☆ | Miniature of Cellarity's platform |
| **P2** Fibrosis-EMT Atlas | ★★★☆☆ | ★★★★★ | ★★★★☆ | Cross-organ fibrosis = NASH pipeline |
| **P3** Foundation Model | ★★★★★ | ★★★☆☆ | ★★★★☆ | scGPT + drug target = cutting-edge |
| **P4** Biomarker Platform | ★★☆☆☆ | ★★★★☆ | ★★★★★ | End-to-end clinical translation |
| **P5** GNN Regulatory | ★★★★★ | ★★★☆☆ | ★★★★☆ | Multi-omics + graph ML = rare skill |
| **P6** Federated Learning | ★★☆☆☆ | ★★★★☆ | ★★★★★ | Owkin-style privacy-preserving ML |

---

## Recommended Execution Order

```
PHASE 1 — Foundation (Weeks 1–5)
├── P1: Cell State Transition Predictor
│   └── Why first: Directly demonstrates Cellarity's thesis, uses your existing data
├── P2: Fibrosis-EMT Atlas (start data download + processing in parallel)
│   └── Why parallel: data-heavy, needs time to download + harmonize 4 datasets
└── Deliverable: 2 working projects, 2 GitHub repos with READMEs

PHASE 2 — Differentiation (Weeks 5–10)
├── P3: Foundation Model Fine-Tuning
│   └── Why: Rare skill, directly relevant to Cellarity + Sanofi
├── P4: Digital Biomarker Platform
│   └── Why: Clinical translation story for Sanofi + Novo Nordisk
└── Deliverable: 4 projects total, interactive demos, bioRxiv preprint draft from P2

PHASE 3 — Advanced (Weeks 10–16)
├── P5: Multi-Omic GNN (if targeting Cellarity or deep comp bio roles)
├── P6: Federated Learning (if targeting Sanofi or data-privacy-sensitive roles)
└── Deliverable: Portfolio complete, 1 preprint submitted, 1 open-source PR merged
```

---

## Interview Story Arcs

### For Cellarity
> "My research focuses on cell state transitions — specifically EMT, the quintessential cell state switch. In my wound healing project, I mapped these transitions at single-cell resolution, then built a perturbation predictor that ranks which gene knockdowns reverse mesenchymal states. I also trained a GNN on multi-omic regulatory networks to identify master regulators. This is exactly your platform's approach — map cell states, understand the regulatory networks, predict chemical interventions."

### For Novo Nordisk
> "I built a cross-organ fibrosis atlas comparing wound healing, NASH, IPF, and CKD at single-cell resolution. The key insight: wound healing and fibrosis share 50 core EMT/ECM genes, but wound healing has 'resolution programs' that fibrosis lacks. These resolution genes are your drug targets — reactivating them could resolve NASH fibrosis. I validated this computationally with survival analysis in TCGA clinical cohorts."

### For Sanofi Digital
> "I built two things relevant to your digital strategy. First, an automated biomarker discovery platform that goes from scRNA-seq signatures to clinical validation — it rediscovered known anti-PD1 biomarkers and identified novel candidates. Second, a federated learning simulation showing that multi-site biomarker discovery works without data exchange — directly relevant to your Owkin partnership. Both demonstrate I can build the computational infrastructure that powers your AI-first R&D."

---

## Key Talking Points by Company

### Cellarity
- "Cell states, not targets" — your EMT work is cell state biology by definition
- "Multi-omics" — show scRNA + scATAC integration
- "Foundation models" — scGPT/Geneformer fine-tuning
- "Lab-in-the-loop" — your perturbation predictions could guide wet lab validation
- Know their pipeline: CLY-124 for Sickle Cell Disease
- Reference: Sophie Tritschler (Comp Bio), Mahdi Zamanighomi (ML) — look up their publications

### Novo Nordisk
- NASH/MASH fibrosis is their next frontier after GLP-1
- Oxford centre: "deep disease understanding + big data" — your atlas demonstrates this
- RNAi (GalXC): your GRN analysis identifies TF targets suitable for RNAi knockdown
- Cambridge MA: external innovation partnerships — frame yourself as a collaborator
- Talk about: resolution vs. persistence in fibrosis — why wound healing resolves but NASH doesn't

### Sanofi Digital
- 3-pillar AI strategy — position yourself in "Expert AI" (deep ML for R&D)
- Owkin partnership — federated learning for precision medicine
- CodonBERT — shows they value foundation model expertise
- Target discovery engines: "7 novel targets in 1 year" — your pipeline does the same thing
- Open Targets consortium — emphasize open-source contributions
- Paris ecosystem: Bioptimus + Sanofi + Owkin = AI-bio hub (network effect)

---

*Created: April 17, 2026*  
*Focus: Hard, high-impact projects only — career-defining portfolio pieces*
