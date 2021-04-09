# Dual Contrastive Learning with GraphCL

## 1 Requirements

`` python>=3.6.2``

`` pytorch==1.4.0``

## 2 Download

#### 2.1 data:
https://drive.google.com/file/d/1bbt7PxYQTRwIkEwcpaH9dyJUwcXFZZVr/view?usp=sharing


#### 2.2 model
https://drive.google.com/file/d/1paLvK53hGWQ2nFeW7fgf1hL3T1qu9cx6/view?usp=sharing

#### 2.3 Usage
Download them and extract them into this directory:
``tar xvf data.tar``
``tar xvf trained_models.tar``

## 3 Command
To run the code:
``./scripts/run.sh``
You may modify the ``dataset``, ``augments``, ``ps`` and ``weights`` in the script accordingly, where 
``datset`` is one of ``cora`` or ``citeseer``,
``augments`` is one of ``node``, ``edge``, ``mask`` and ``subgraph``.
``ps`` is the modification rate of the augmentation, 
``weights`` are values of $ \lambda $ mentioned in the section 5 of the report.

## Acknowledgement
This code is built upon the implementation of https://github.com/Shen-Lab/GraphCL



