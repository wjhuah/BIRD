# BIRD: Bronze Inscription Restoration and Dating  

> 《詩·商頌·玄鳥》：天命玄鳥，降而生商  
> *“Heaven commissioned the swallow, to descend and give birth to Shang.”*  
> — translated by James Legge

[![Hugging Face Model](https://img.shields.io/badge/🤗%20Model-SikuRoBERTa_Bronze-yellow)](https://huggingface.co/wjhuah/SikuRoBERTa_Bronze)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-BIRD-blue)](https://huggingface.co/datasets/wjhuah/BIRD)

---

## Data

BIRD consists of encoded bronze inscription texts and paleographic resources
supporting both restoration and dating.

- **Domain corpus** — `data/dapt.txt`  
  Pre-Qin texts for domain-adaptive pretraining (by linguistic register).  
  *Example:* 絕智棄辯，民利百倍。絕巧棄利，盜賊亡有。絕偽棄慮，民復季子。
  
- **Restoration** — `data/tapt_*.txt`  
  Plain inscription text used for masked language modeling.  
  *Example:* 唯十又九年，四月既朢辛卯，王在周康卲宮，各于大室，即位……

- **Dating** — `data/dating.csv`  
  Texts annotated with dynasty and period labels.  
  *Example:* 春秋｜晚期｜秦王俾命競墉，王之定，救秦戎。

- **Glyph Net** — `data/*.edge`  
  Allograph relations encoding glyph variants.  
  *Example:* 丌 → 其
  
---

## Overview

Bronze inscriptions from early China are often fragmentary and difficult to date.
We introduce **BIRD** (**B**ronze **I**nscription **R**estoration and **D**ating),
a fully encoded dataset and modeling framework grounded in standard scholarly
transcriptions and chronological labels.

BIRD formulates restoration as an **allograph-aware masked language modeling**
problem and integrates domain-adaptive pretraining (DAPT), task-adaptive
pretraining (TAPT), and a **Glyph Net (GN)** that links graphemes with their
allographs. Experiments show consistent gains in both restoration and
chronological dating.

<p align="center">
  <img src="figure/pipeline.png" width="90%">
</p>

> **BIRD pipeline.**  
> DAPT and TAPT are combined with a Glyph Net to inject allograph-level structure
> into a BERT/RoBERTa backbone, supporting both restoration and dating.

---

## Repository Structure

```text
.
├── main.py            # experimental pipeline
├── data/              # input data and glyph net
├── results/           # model evaluation outputs
├── errors/            # mis-predictions
├── UNK/               # unknown glyphs
├── figure/            
├── README.md
└── LICENSE
````
---

## Paleographical Reference

[Document Link](https://docs.google.com/document/d/11RvdpOuW2UnSzYNukp-jEOZJsN5NNkzX/edit?usp=sharing&ouid=110717833428221317700&rtpof=true&sd=true)

*Last update: Jun 29, 2026*

---

## Citation

If you find this useful, please cite our paper:

```bibtex
@inproceedings{hua2025bird,
  title     = {BIRD: Bronze Inscription Restoration and Dating},
  author    = {Hua, Wenjie and Nguyen, Hoang H. and Ge, Gangyan},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  year      = {2025},
  publisher = {Association for Computational Linguistics}
}
```

