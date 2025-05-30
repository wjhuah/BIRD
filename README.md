# Bronze Inscription Restoration

This project establishes the first fully Unicode-encoded database for modeling and restoring bronze inscriptions from early China. It supports multiple sub-tasks:

- **Masked Language Modeling (MLM)** for missing character prediction
- **Chronological Classification** for periodization of bronze texts

Key contributions include:

- A fully digitized and normalized corpus of over 40,000 inscriptional characters
- A glyph network aligning diachronic variants to semantic anchors
- A variant-aware MLM framework with domain- and task-adaptive pretraining (DAPT + TAPT)

## 🔹 Corpus Statistics

| Type                | Count   | Proportion |
|---------------------|---------|------------|
| Identifiable        | 39,565  | 99.24%     |
| Ambiguous (□)       | 236     | 0.59%      |
| Unknown ([UNK])     | 56      | 0.14%      |

## 🔹 MLM Results

| Model       | DAPT | GN       | Top-1 / Top-10 |
|-------------|------|----------|----------------|
| mBERT       | --   | replace  | .4580 / .6270  |
|             | --   | inject   | .4797 / .6531  |
|             | +    | replace  | .4570 / .6378  |
|             | +    | inject   | **.4833 / .6580** |
| SIKU-BERT   | --   | replace  | .5049 / .6871  |
|             | --   | inject   | .5353 / .7290  |
|             | +    | replace  | .5111 / .7012  |
|             | +    | inject   | **.5420 / .7263** |

## 🔹 Perplexity (PPL)

| Scenario                   | mBERT PPL | SIKU-BERT PPL |
|---------------------------|-----------|---------------|
| Baseline                  | 169.39    | 1253.66       |
| TAPT + GN (inject)        | 16.14     | 14.05         |
| TAPT + GN (replace)       | 18.27     | 18.25         |
| DAPT + TAPT + GN (inject) | **15.98** | **13.86**     |
| DAPT + TAPT + GN (replace)| 17.48     | 17.15         |

## 🔹 Periodization Accuracy

| Classifier         | Dynasty Acc | Period Acc (avg) |
|--------------------|-------------|------------------|
| Logistic Regression| .7407       | .5003            |
| Naive Bayes        | .7222       | .5371            |
| Linear SVM         | **.7840**   | **.5932**        |
| Random Forest      | .7747       | .5661            |

## 🔹 Project Files

```
├── data/
│   └── dyn_br.json              # Annotated with dynasty and period labels
├── mlm/
│   ├── dapt.txt                 # Domain-adaptive pretraining data
│   ├── train_inj.txt            # Injection-style variant training data
│   ├── test_inj.txt
│   ├── train_rep.txt            # Replacement-style training data
│   └── test_rep.txt
├── glyph/
│   ├── Yin.xlsx
│   ├── Western_Zhou.xlsx
│   └── SpringAutumn_WarringStates.xlsx
├── main.py                      # Main script for MLM training
├── dyn_main.py                 # Script for dynasty/period classification
├── LICENSE
└── README.md
```

## 🔹 License

All data and scripts are released under the [MIT License](./LICENSE). For academic use or citation, please contact the authors.

## 🔹 Citation

*For full methodology and experiments, see the accompanying paper.*
