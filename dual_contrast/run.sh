seeds=(19)
augments=("subgraph")
ps=(0.2)
weights=(0.7)
set -e
dataset="citeseer"
for seed in ${seeds[@]}; do
for aug in ${augments[@]}; do
for p in ${ps[@]}; do
for w in ${weights[@]}; do
name=${dataset}_${aug}_${p}_${w}_${seed}
touch ${name}.txt
CUDA_VISIBLE_DEVICES=$1 python -u execute.py --dataset ${dataset} --aug_type ${aug} --drop_percent ${p} --seed ${seed} --save_name ${name}.pkl --gpu 0 --weight ${w} # > ${name}.txt
# CUDA_VISIBLE_DEVICES=$1 python -u execute.py --dataset citeseer --aug_type ${aug} --drop_percent ${p} --seed ${seed} --save_name cite_best_dgi.pkl --gpu 0 --weight ${w}
done done done done