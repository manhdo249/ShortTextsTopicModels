export WANDB_API_KEY="c404830b3fe76c9bae6be1dc53effe3226b28175"

MODEL_NAME=FASTopic
NUM_TOPICS=100
DATASET=SearchSnippetsCluster
GLOBAL_DIR=umap_globalcluster60

WANDB_PROJECT=FASTopic_100topics_SearchSnippets

for seed in $(seq 0 10)
do
    python main.py  \
        --model $MODEL_NAME \
        --num_topics $NUM_TOPICS \
        --dataset $DATASET \
        --seed $seed \
        --wandb_on \
        --wandb_prj $WANDB_PROJECT \
        --wandb_name "${MODEL_NAME}_${DATASET}_top${NUM_TOPICS}_seed${seed}" \
        --verbose 
done