export WANDB_API_KEY="c404830b3fe76c9bae6be1dc53effe3226b28175"

MODEL_NAME=KNNTM
NUM_TOPICS=50
DATASET=GoogleNewsCluster
GLOBAL_DIR=umap_globalcluster150

WANDB_PROJECT=KNNTM_50topics_GoogleNews_only_theta
alpha=1.0
num_k=20

for seed in $(seq 0 2)
do
    python main.py  \
        --model $MODEL_NAME \
        --alpha $alpha \
        --num_k $num_k \
        --eta 1.0 \
        --num_topics $NUM_TOPICS \
        --dataset $DATASET \
        --p_epochs 20 \
        --seed $seed \
        --wandb_on \
        --wandb_prj $WANDB_PROJECT \
        --wandb_name "${MODEL_NAME}_${DATASET}_top${NUM_TOPICS}_alpha${alpha}_num_k${num_k}_only_theta_seed${seed}" \
        --verbose
done