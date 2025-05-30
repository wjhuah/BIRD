# Bronze Inscription Restoration

This project aims to establish the first fully Unicode-based corpus for training models on bronze inscriptions and supporting downstream tasks. We conduct initial experiments on:

- **Masked Language Model (MLM) for missing character prediction**
- **Dynasty classification and period prediction from bronze inscriptions**

Our philological reliability is supported by references such as *Jinwen Morphology Studies* and *Deep Learning-Based Analysis of Pre-Qin Literature*.

## 🔹 Project Structure

```
├── data/
│   └── dyn_br.json              # Annotated texts with dynasty and period labels
├── mlm/
│   ├── dapt.txt                 # Domain-adaptive pretraining corpus
│   ├── train_inj.txt            # Injection-style training data
│   ├── test_inj.txt             # Injection-style test data
│   ├── train_rep.txt            # Replacement-style training data
│   └── test_rep.txt             # Replacement-style test data
├── glyph/
│   ├── Yin.xlsx
│   ├── Western_Zhou.xlsx
│   └── SpringAutumn_WarringStates.xlsx
├── main.py                      # MLM training script
├── dyn_main.py                  # Dynasty/period classification script
├── LICENSE
└── README.md
```

## 🔹 Usage

### 1. MLM Training & Evaluation

```bash
python main.py
```

This runs training under multiple configurations and outputs `ablation_all.csv`:

- Baseline
- DAPT + TAPT
- Inject/Replace glyph variant strategies

### 2. Dynasty & Period Classification

```bash
python dyn_main.py
```

Input is `dyn_br.json` (or CSV), output shows accuracy across hierarchy levels.

| Classifier           | Dynasty Acc | Period Acc (avg) |
|----------------------|-------------|------------------|
| Logistic Regression  | 74.07%      | 50.03%           |
| Naive Bayes          | 72.22%      | 53.71%           |
| Linear SVM           | **78.40%**  | **59.32%**       |
| Random Forest        | 77.47%      | 56.61%           |

## 🔹 License

All corpora and toolchains in this project are original contributions, released under the [MIT License](./LICENSE). For citation or academic use, please contact the author.

## 🔹 Reference Summary

| Type         | Count   | Proportion |
|--------------|---------|------------|
| Identifiable | 39,565  | 99.24%     |
| Ambiguous (□) | 236     | 0.59%      |
| Unknown ([UNK]) | 56    | 0.14%      |

| Model     | DAPT | GN     | Top-1 / -10 |
|-----------|------|--------|-------------|
| mBERT     | –    | replace| .4580 / .6270 |
|           | –    | inject | .4797 / .6531 |
|           | +    | replace| .4570 / .6378 |
|           | +    | inject | .4833 / .6580 |
| SIKU-BERT | –    | replace| .5049 / .6871 |
|           | –    | inject | .5353 / .7290 |
|           | +    | replace| .5111 / .7012 |
|           | +    | inject | .5420 / .7263 |

| Classifier           | Dynasty | Period (avg) |
|----------------------|---------|--------------|
| Logistic Regression  | .7407   | .5003         |
| Naive Bayes          | .7222   | .5371         |
| Linear SVM           | .7840   | .5932         |
| Random Forest        | .7747   | .5661         |
