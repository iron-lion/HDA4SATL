# HDA4SATL

Analysis code for [**"Species-Agnostic Transfer Learning for Cross-species Transcriptomics Data Integration without Gene Orthology"**](https://doi.org/10.1093/bib/bbae004).

Cross-species comparison of single-cell RNA-seq data is challenging because gene sets differ between species, making direct alignment non-trivial. HDA4SATL frames the problem as **Generalized Zero-Shot Learning (GZSL)**: the source species provides labeled cell-type annotations, and the model must classify target-species cells — including cell types never seen during training — without relying on gene orthology mappings.


## Repository structure

```
HDA4SATL/
└── hda_for_satl/
    ├── src/
    │   ├── satl.py            # SATL orchestrator (main model)
    │   ├── model_cada.py      # CADA-VAE implementation
    │   ├── data_loader.py     # Dataset loading & preprocessing
    │   ├── final_classifier.py# GZSL/ZSL classifier & H-score evaluation
    │   ├── utils.py           # Metrics, sampling, visualisation helpers
    │   └── model_block.py     # Shared neural network building blocks
    ├── scenario_satl_baron.py            # SATL — Baron pancreas dataset
    ├── scenario_satl_mouse_to_human.py   # SATL — Mouse→Human organ datasets
    ├── scenario_satl_lps_stimul.py       # SATL — LPS stimulation data
    ├── scenario_satl_mouse_to_human_external.py
    ├── scenario_cada_mouse_to_human.py   # CADA-VAE — Mouse→Human
    ├── scenario_mnn_baron.py             # MNN — Baron dataset
    ├── scenario_mnn_lps_stimul.py        # MNN — LPS stimulation
    ├── scenario_adapt_baron.py           # TCA — Baron dataset
    └── dataset/                          # Preprocessed datasets
        ├── baron_norm/                   # Baron pancreas (mouse & human)
        ├── bm/, brain/, pancreas/        # Mouse-Human organ pairs
        └── norm_pca50/                   # PCA-reduced data
```

## Key concepts

**Generalized Zero-Shot Learning (GZSL) setup** — At evaluation time the model encounters both *seen* cell types (present in source training data) and *unseen* cell types (masked out during training). Performance is measured with the **H-score**: the harmonic mean of accuracy on seen and unseen classes.

**Masking protocol** — Each experimental run randomly masks `n_missing` cell types from the target domain, trains on the remainder, and predicts on all. Results are averaged across all valid masking combinations.

**Semi-supervised extension** — After an initial prediction pass, high-confidence target predictions are added as pseudo-labels and used for a second CDSPP fitting, improving coverage of unseen classes.

## Quickstart

```python
from hda_for_satl.src.satl import SATL
from hda_for_satl.src.data_loader import baron_load
from sklearn.decomposition import PCA

# Load dataset
dataset, label_set = baron_load('./hda_for_satl/dataset/baron_norm/', PCA(n_components=50))

# Configure and run SATL
model = SATL(
    dataset,
    out_dir='./results/',
    n_missing=2,          # number of cell types to mask per run
    dim=11,               # CDSPP latent dimension
    alpha=[0.01, 0.1, 1, 10, 100],  # regularisation candidates (CV-selected)
    K=10,                 # KNN neighbours
    n_jobs=4,             # parallel workers
)
model.run_mode()          # writes H-scores and predictions to out_dir
```

Or run a pre-configured scenario directly:

```bash
cd hda_for_satl
python scenario_satl_baron.py
```

## Dependencies

Install the required **CDSPP** library:

```bash
git clone https://github.com/iron-lion/cdspp-hda-py
cd cdspp-hda-py
pip install .
```

Other dependencies: `scikit-learn`, `imbalanced-learn`, `torch`, `anndata`, `h5py`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`.



## Citation

If you use this code please cite:

```
Park, Youngjun, Nils P. Muttray, and Anne-Christin Hauschild.
"Species-agnostic transfer learning for cross-species transcriptomics
data integration without gene orthology."
Briefings in Bioinformatics 25.2 (2024): bbae004.
```

```bibtex
@article{park2024species,
  title   = {Species-agnostic transfer learning for cross-species transcriptomics data integration without gene orthology},
  author  = {Park, Youngjun and Muttray, Nils P. and Hauschild, Anne-Christin},
  journal = {Briefings in Bioinformatics},
  volume  = {25},
  number  = {2},
  pages   = {bbae004},
  year    = {2024},
  doi     = {10.1093/bib/bbae004}
}
```

## Acknowledgements

- [CDSPP original MATLAB implementation](https://github.com/hellowangqian/cdspp-hda)
- [Master thesis work (het_GZSDA)](https://github.com/nmuttray/het_GZSDA/tree/main)
