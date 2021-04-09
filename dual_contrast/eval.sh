dataset="cora"
augments=("edge")
set -e
for aug in ${augments[@]}; do
name=${dataset}_${aug}
python -u eval.py --dataset ${dataset} --aug ${aug} --save_name models2upload/${name}.pkl --gpu 0 --seed 19 
done