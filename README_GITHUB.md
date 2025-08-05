# AdaRocile: Ensemble Reconciliation Framework

**Supplementary Material for AAAI 2026 Conference Submission**

This repository contains the complete implementation of the **Rocile** and **AdaRocile** ensemble reconciliation frameworks, along with comprehensive evaluation on 7 real-world datasets.

## 🎯 Overview

This package implements two novel ensemble reconciliation algorithms:

- **Rocile**: Momentum-based batch sequential reconciliation with adaptive learning rates
- **AdaRocile**: Local bias correction + Rocile reconciliation with 5 different patching strategies

## 📊 Datasets

### Classification Datasets (6)
- **German** (47KB) - Credit risk classification
- **COMPAS** (2.2MB) - Recidivism prediction  
- **Adult** (2.4MB) - Income classification
- **Folk Mobility** (1.7MB) - Mobility prediction
- **Folk Travel** (4.2MB) - Travel time classification
- **Folk Income** (11.7MB) - Income prediction

### Regression Dataset (1)
- **Communities** (974KB) - Crime rate regression

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Run setup script
chmod +x setup.sh
./setup.sh
```

### Basic Usage
```python
from mypackage.model import Rocile, AdaRocile

# Rocile for basic reconciliation
rocile = Rocile(random_state=42)
rocile.fit(X_train, y_train)
predictions = rocile.predict(X_test)

# AdaRocile with local patching
adarocile = AdaRocile(patch_strategy='BiasCorrected', random_state=42)
adarocile.fit(X_train, y_train)
predictions = adarocile.predict(X_test)
```

### Run Individual Dataset Experiments
```bash
# Run German dataset experiment
cd Examples/German_Data
python3 run_german_dataset.py

# Run all experiments
python3 run_all_examples.py
```

## 🔬 Local Patching Strategies

AdaRocile supports 5 different local patching strategies:

1. **BiasCorrected** (default): Error direction analysis
2. **DistanceWeighted**: Inverse distance weighting
3. **ModelSpecific**: Model-specific averaging
4. **CertaintyWeighted**: Inverse variance weighting
5. **EnsembleLevel**: Ensemble-level correction

## 📈 Key Results

### Ensemble Disagreement Reduction
- **German Dataset**: 100% disagreement reduction (0.7300 → 0.0000)
- **COMPAS Dataset**: 100% disagreement reduction (0.3666 → 0.0000)

### Performance Improvements
- Consistent accuracy improvements across all datasets
- Significant reduction in ensemble variance and ambiguity
- Enhanced reliability and calibration

## 🏗️ Package Structure

```
├── mypackage/
│   ├── __init__.py          # Package initialization
│   └── model.py             # Core Rocile and AdaRocile implementations
├── datasets/                # All 7 research datasets
├── Examples/                # Individual dataset runners
│   ├── German_Data/
│   ├── Compas_Data/
│   ├── Communities_Data/
│   ├── Adult_Data/
│   ├── Folk_Mobility_Data/
│   ├── Folk_Travel_Data/
│   └── Folk_Income_Data/
├── tests/                   # Comprehensive test suite
├── utils.py                 # Utility functions
├── train.py                 # Basic training demonstration
├── evaluate.py              # Detailed evaluation
├── run_all_examples.py      # Run all datasets with all strategies
├── test_patch_strategies.py # Test all local patching strategies
├── run_tests.py             # Custom test runner
├── setup.py                 # Package installation
├── requirements.txt         # Dependencies
├── setup.sh                 # Automated setup
└── README.md                # Comprehensive documentation
```

## 🧪 Testing

```bash
# Run all tests
python3 run_tests.py

# Test patch strategies
python3 test_patch_strategies.py

# Verify submission
python3 verify_submission.py
```

## 📋 Requirements

- Python 3.8+
- scikit-learn >= 1.2
- pandas
- numpy
- matplotlib
- DESlib

## 📚 Documentation

- **README.md**: Complete usage guide and API documentation
- **SUBMISSION_SUMMARY.md**: Comprehensive submission overview
- **Code Comments**: Detailed inline documentation

## 🔍 Verification

Run the verification script to ensure all components are working:
```bash
python3 verify_submission.py
```

## 📄 Citation

If you use this code in your research, please cite our AAAI 2026 paper:

```bibtex
@inproceedings{adarocile2026,
  title={AdaRocile: Adaptive Local Patching for Ensemble Reconciliation},
  author={Anonymous},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

## 📞 Contact

For questions about this implementation, please refer to the paper or contact the authors through the conference submission system.

---

**Note**: This is supplementary material for AAAI 2026 conference submission. The repository is maintained anonymously for the review process. 