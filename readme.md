# Linear regression for sentence semantic similarity
Implementation of Linear regression + embeddings for sentence semantic similarity

### Contents

* [Installation](#installation)
* [Usage](#usage)
  * [Train](#train)

---

## Installation
```
virtualenv venv -p python3
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Train linear regression + embeddings models

Language: br (PT-BR) or pt (PT-EU)

```
python evaluate_assin2.py ./models/<pretrained embedding> <language>
```


Scripts for ASSIN
=================

This repository contains Python scripts used in the `ASSIN`_ shared task. There is the evaluation script, ``assin-eval.py``, and the two baseline scripts, ``baseline-overlap.py`` and ``baseline-majority.py``. 

.. _ASSIN: http://nilc.icmc.usp.br/assin/

In order to learn how to use them, just call any of them in the command line with the ``-h`` option.
