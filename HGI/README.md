# Hierarchical Graph Infomax

Author: Li Yuan A0198759Y

Partially referenced from: [https://github.com/fanyun-sun/InfoGraph](https://github.com/fanyun-sun/InfoGraph) and [https://github.com/cszhangzhen/HGP-SL](https://github.com/cszhangzhen/HGP-SL)

## Search hyperparameter (optional)

`nnictl create --config nniconf/<conf>.yml --port <port>`

For example, `nnictl create --config nniconf/config_proteins.yml --port 8088`

This step is optional. We provide searched hyperparameters in `parallel_run.sh`

## Train from scratch

`bash parallel_run.sh <dataset> savef <cuda device IDs>`

For example, `bash parallel_run.sh MUTAG savef 0 1 2 3 4`

This command runs 5 experiments simultaneously. The datasets are automatically downloaded by pytorch-geometric during training. The checkpoints and training logs are saved in `ckpts/` and `logs/`, respectively.

## Evaluate checkpoints

`bash parallel_run.sh <dataset> loadf <cuda device IDs>`

For example, `bash parallel_run.sh MUTAG loadf 0 1 2 3 4`

This command evaluates 5 checkpoints simultaneously. The evaluation logs are saved in `logs/`.

Before evaluation, please ensure trained models are placed in `/ckpts`, which are either trained from scratch or downloaded from our [Google Drive link](https://drive.google.com/drive/folders/1UJFaY88ANnScjGvET6AWx-1QmgdN1yZE?usp=sharing). 

Note that due to random K-fold evaluation, the results can be different from the training logs.
