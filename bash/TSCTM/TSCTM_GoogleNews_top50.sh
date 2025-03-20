export WANDB_API_KEY="c404830b3fe76c9bae6be1dc53effe3226b28175"

MODEL_NAME=TSCTM
NUM_TOPICS=50
DATASET=GoogleNewsCluster
GLOBAL_DIR=umap_globalcluster150

WANDB_PROJECT=TSCTM_50topics_GoogleNews

for weight_contrast in 1.0
do
    for temperature in 0.5
    do
        for seed in $(seq 0 2)
        do 
            echo 2 | CUDA_VISIBLE_DEVICES=0 python main.py  \
                --model $MODEL_NAME \
                --weight_contrast $weight_contrast \
                --temperature $temperature \
                --num_topics $NUM_TOPICS \
                --dataset $DATASET \
                --seed $seed \
                --wandb_on \
                --wandb_prj $WANDB_PROJECT \
                --wandb_name "${MODEL_NAME}_${DATASET}_top${NUM_TOPICS}_weight_contrast${weight_contrast}_temperature${temperature}_seed${seed}" \
                --verbose
        done
    done
done
