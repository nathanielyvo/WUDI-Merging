
# set -e pipefail

date_today=$(date '+%Y-%m-%d')
outdir=${outdir:="outs/merge_results"}
mkdir -p ${outdir}


models_name=(
"cola"
"sst2"
"mrpc"
"stsb"
"qqp"
"mnli"
"qnli"
"rte"
)

models_to_merge=()
for d in "${models_name[@]}"; do
# models_to_merge+=(../roberta/$d/roberta-base_lr1e-05)
models_to_merge+=(../../MergeLM/save_models/$d/roberta-large_lr1e-05)
done
select_merge=${select_merge:="8"}


function pos(){

if [ $select_merge -eq 1 ]; then
    echo "please set \$select_merge > 1"
    exit 1 
fi
src_merge=("${models_name[@]:0:$select_merge}") 

echo ">>> merged from $select_merge tasks"
echo ">>> merge ${src_merge[@]}"

data_path="data/test.json"
}


function run_dare_task_arith(){

pos

for i in 0.7 0.8 0.9; do

python run_merge.py \
--models-to-merge ${models_to_merge[@]} \
--models-name ${models_name[@]} \
--src-merge ${src_merge[@]} \
--data-path $data_path \
--yaml-file config/dare_merge.yml \
--exclude-param ".*classifier.*" ".*bias.*"  \
--base-model 'roberta-large' \
--mask-rate $i \
--outdir $outdir

done

}

function run_dare_tie(){

pos

for i in 0.7 0.8 0.9; do

python run_merge.py \
--models-to-merge ${models_to_merge[@]} \
--models-name ${models_name[@]} \
--src-merge ${src_merge[@]} \
--data-path $data_path \
--yaml-file config/dare_merge2.yml \
--exclude-param ".*classifier.*" ".*bias.*"  \
--base-model 'roberta-large' \
--mask-rate $i \
--outdir $outdir

done

}


function run_avg_merge(){

pos

python run_merge.py \
--models-to-merge ${models_to_merge[@]} \
--models-name ${models_name[@]} \
--src-merge ${src_merge[@]} \
--data-path $data_path \
--yaml-file config/average_merge.yml \
--exclude-param ".*classifier.*" ".*bias.*"  \
--base-model 'roberta-large' \
--outdir $outdir


}

function run_ta_decompose(){

pos
# for i in 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35; do
# for l1_coef in 0.0005 0.00007 0.001 0.003 0.005 0.007 0.01 0.03 0.05 0.07 0.1 0.3 0.5 0.7 1.0; do
for i in 0.8; do
# for l1_coef in 0 0.00001 0.0001 0.001 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 1; do
for l1_coef in 1; do
python run_merge.py \
--models-to-merge ${models_to_merge[@]} \
--models-name ${models_name[@]} \
--src-merge ${src_merge[@]} \
--data-path $data_path \
--yaml-file config/task_arithmetic_decompose.yml \
--exclude-param ".*classifier.*" ".*bias.*" ".*LayerNorm.*" ".*embeddings.*" \
--scaling $i \
--outdir $outdir \
--save-path "outs/task_arithmetic_decompose" \
--l1-coef $l1_coef \
--base-model Roberta-large
done
done
}

function run_tie(){

pos


for i in 0.9; do
for j in 0.7; do

python run_merge.py \
--models-to-merge ${models_to_merge[@]} \
--models-name ${models_name[@]} \
--src-merge ${src_merge[@]} \
--yaml-file config/ties_merge.yml \
--data-path $data_path \
--exclude-param ".*classifier.*" ".*bias.*"  \
--mask-rate $i \
--base-model 'roberta-large' \
--scaling $j \
--outdir $outdir

done
done

}


function run_task_arith(){

pos


for j in 0.8; do

python run_merge.py \
--models-to-merge ${models_to_merge[@]} \
--models-name ${models_name[@]} \
--src-merge ${src_merge[@]} \
--data-path $data_path \
--yaml-file config/task_arithmetic.yml \
--exclude-param ".*classifier.*" ".*bias.*"  \
--scaling $j \
--base-model 'roberta-large' \
--outdir $outdir \
--save-path "outs/task_arithmetic"

done

}
function ft_decompose(){

pos

python run_merge.py \
--models-to-merge ${models_to_merge[@]} \
--models-name ${models_name[@]} \
--src-merge ${src_merge[@]} \
--base-model 'roberta-large' \
--data-path $data_path \
--exclude-param ".*classifier.*" ".*bias.*" ".*LayerNorm.*" ".*embeddings.*" \
--yaml-file config/ft_decompose.yml \
--outdir "outs/finetuned_decompose" 

}
# --exclude-param ".*classifier.*" \
# --exclude-param ".*classifier.*" ".*bias.*" ".*LayerNorm.*" ".*embeddings.*" \


function ft(){

pos

python run_merge.py \
--models-to-merge ${models_to_merge[@]} \
--models-name ${models_name[@]} \
--src-merge ${src_merge[@]} \
--base-model 'roberta-large' \
--data-path $data_path \
--exclude-param ".*classifier.*" \
--outdir "outs/finetuned" 

}

function pretrain(){

pos

python run_merge.py \
--models-to-merge 'NONE' \
--models-name 'NONE' \
--src-merge ${src_merge[@]} \
--data-path $data_path \
--base-model 'roberta-large' \
--outdir $outdir 

}


function twin_merge(){

yml='config/twin_merge.yml'
# NOTICE: we only select prefix 
select_merge=${select_merge:="8"}
select_twin=${select_twin:="8"}

if [ $select_merge -eq 1 ]; then
    echo "please set \$select_merge > 1"
    exit 1 
elif [ $select_twin -eq 1 ]; then
    datapath="data_glue/new_dataset2.json"
    if [ -z $src_twin ];then
        echo "please set \$src_twin!"
        exit 1
    fi
else
    datapath=data/test_router.json
    src_twin=("${models_name[@]:0:$select_twin}") 
    src_merge=("${models_name[@]:0:$select_merge}") 
fi

mask_strategy=${mask_strategy:="svd"}
mask_rate=${mask_rate:="0.9"}
echo ">>> use data_path $datapath"
echo ">>> use outdir $outdir"
echo ">>> merged from $select_merge tasks"
echo ">>> use twin vector from $select_twin tasks"
echo ">>> mask_rate $mask_rate; mask_strategy $mask_strategy"
echo ">>> use yml $yml"

python twin_merge.py \
--models-to-merge ${models_to_merge[@]} \
--models-name ${models_name[@]} \
--data-path $datapath \
--src-merge ${src_merge[@]} \
--src-twin ${src_twin[@]} \
--yaml-file $yml \
--share-expert outs/task_arithmetic \
--exclude-param ".*classifier.*" ".*bias.*" \
--mask-rate $mask_rate \
--mask-strategy $mask_strategy \
--outdir $outdir 

}
