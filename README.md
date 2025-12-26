# BIRD: Bronze Inscription Restoration and Dating  

> 《詩·商頌·玄鳥》：天命玄鳥，降而生商  
> *“Heaven commissioned the swallow, to descend and give birth to Shang.”*  
> — translated by James Legge

[![Hugging Face Model](https://img.shields.io/badge/🤗%20Model-SikuRoBERTa_Bronze-yellow)](https://huggingface.co/wjhuah/SikuRoBERTa_Bronze)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-BIRD-blue)](https://huggingface.co/datasets/wjhuah/BIRD)

---

## Overview

Bronze inscriptions from early China are often fragmentary and difficult to date.
We introduce **BIRD** (**B**ronze **I**nscription **R**estoration and **D**ating),
a fully encoded dataset grounded in standard scholarly transcriptions and
chronological labels.

BIRD further proposes an **allograph-aware masked language modeling framework**
that integrates:
- domain-adaptive pretraining (DAPT),
- task-adaptive pretraining (TAPT),
- and a Glyph Net (GN) linking graphemes and allographs.

Experiments show that GN improves character restoration, while glyph-biased sampling yields consistent gains in chronological dating.

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
