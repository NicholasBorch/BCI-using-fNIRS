# BCI using fNIRS
This repository is the product of the course "02466 Project Work - Bachelor of Artificial Intelligence and Data" at the Technical University of Denmark.
Functional Near-Infrared Spectroscopy (fNIRS) analysis for brain–computer interface (BCI) research.  
The repository bundles **pre-processing**, **feature extraction**, **unsupervised Gaussian-mixture modelling (GMM)** sanity checks, and an automated **random feature-subset search** aimed at maximising F₂-score in Control vs Task discrimination.

---

## Repository Structure

```
├── finger_tapping/
│   ├── preprocessing.py          # Epoching, band-pass filtering, ICA helpers
│   ├── feature_preparation.py    # 30-feature extractor (raw + ICA-based)
│   ├── search.py                 # Build subject-level tables & GMM evaluation
│   └── ...
├── demo_scripts/                 # Reproducible experiments
│   ├── clinical_sanity.py        # Monte-Carlo split sanity check
│   ├── proportion_sanity.py      # 2-scenario contingency-table check
│   └── random_search.py          # 30,000-trial random feature search
├── search_results/               # Saved outputs (created at runtime)
│   └── all_trials.csv
├── requirements.txt
└── README.md                     # ← you are here
```

---

## Pipelines & Key Scripts

### Feature Extraction
- `simple_pipeline(subj)` – Load preprocessed Control/Left/Right epochs
- `extract_all_epoch_features()` – Extracts 30 features (amplitude, shape, ICA, power)

### Sanity Checks
- `clinical_sanity.py` – Monte Carlo splits of Control-only epochs, chi² tests for label consistency
- `proportion_sanity.py` – Scenario-based GMM fits (Control split vs Control+Task), contingency tables

### Model Evaluation
- `search.py` – GMM clustering and metric evaluation across baseline/ICA/optimal feature sets
- `_fit_gmm_metrics()` – Returns F₂, precision, recall, accuracy, ARI, BIC, silhouette, etc.

### Random Feature Subset Search
- `random_search.py` – Tests 30,000 subsets (≤5 features) to maximize mean F₂-score
- Forced evaluation of known good subsets (e.g., ICA-only, baseline-only)
- Results saved to `search_results/all_trials.csv`

---

## Installation

```bash
git clone https://github.com/NicholasBorch/BCI-using-fNIRS.git
cd BCI-using-fNIRS
pip install . 
pip install -r requirements.txt
```

---

## Output

- Console: F₂ score, accuracy, precision, recall, etc. per feature set
- `search_results/all_trials.csv`: full trial log of feature subset performance
- `Optimal.csv`: best subsets for further evaluation and significance testing

---

## License & Citation

This codebase is open for academic and research use.  
If you use this repository, please cite it or mention:

**https://github.com/NicholasBorch/BCI-using-fNIRS**
