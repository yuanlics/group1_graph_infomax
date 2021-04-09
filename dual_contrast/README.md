# Dual Contrastive Learning with GraphCL


## 1 Requirements

`` python>=3.6.2``

`` pytorch==1.4.0``

## 2 Download

#### 2.1 data:
https://drive.google.com/file/d/1bbt7PxYQTRwIkEwcpaH9dyJUwcXFZZVr/view?usp=sharing


#### 2.2 model
https://drive.google.com/file/d/1LGMHIoFjgTm78_aGNA_dtGuW_5BJHqup/view?usp=sharing

#### 2.3 Usage
Download them and extract them into this directory:
``tar xvf data.tar``
``tar xvf trained_models.tar``

## 3 Command

### 3.1 Training
To run the code:
``./scripts/run.sh``
You may modify the ``dataset``, ``augments``, ``ps`` and ``weights`` in the script accordingly, where 
``datset`` is one of ``cora`` or ``citeseer``,
``augments`` is one of ``node``, ``edge``, ``mask`` and ``subgraph``.
``ps`` is the modification rate of the augmentation, 
``weights`` are values of $ \lambda $ mentioned in the section 5 of the report.

### 3.2 Evaluation
To eval with a trained model, run ``./scripts/run.sh``.

You may adjust the parameters ``dataset`` ("cora" "citeseer") and ``augmentation`` ("node" "edge" "mask" "subgraph") accordingly.

## Acknowledgement
This code is built upon the implementation of https://github.com/Shen-Lab/GraphCL




