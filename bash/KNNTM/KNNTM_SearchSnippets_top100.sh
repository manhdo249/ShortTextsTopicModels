export WANDB_API_KEY="c404830b3fe76c9bae6be1dc53effe3226b28175"

MODEL_NAME=KNNTM
NUM_TOPICS=100
DATASET=SearchSnippetsCluster
GLOBAL_DIR=umap_globalcluster60

WANDB_PROJECT=KNNTM_100topics_SearchSnippets
alpha=1.0
num_k=30
eta=0.2
rho=0.6

for seed in $(seq 0 2)
do
    python main.py  \
        --model $MODEL_NAME \
        --alpha $alpha \
        --num_k $num_k \
        --eta $eta \
        --rho $rho \
        --num_topics $NUM_TOPICS \
        --dataset $DATASET \
        --p_epochs 20 \
        --seed $seed \
        --wandb_on \
        --wandb_prj $WANDB_PROJECT \
        --wandb_name "${MODEL_NAME}_${DATASET}_top${NUM_TOPICS}_alpha${alpha}_num_k${num_k}_eta${eta}_rho${rho}_seed${seed}" \
        --verbose
done