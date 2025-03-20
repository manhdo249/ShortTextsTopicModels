export WANDB_API_KEY="c404830b3fe76c9bae6be1dc53effe3226b28175"

MODEL_NAME=ECRTM
NUM_TOPICS=100
DATASET=GoogleNewsCluster
GLOBAL_DIR=umap_globalcluster150

WANDB_PROJECT=ECRTM_100topics_GoogleNews


for weight_loss_ECR in 30.0 40.0 20.0 10.0 50.0
do
    for seed in $(seq 0 3)
    do 
        CUDA_VISIBLE_DEVICES=0 python main.py  \
                            --model $MODEL_NAME \
                            --weight_loss_ECR $weight_loss_ECR \
                            --pretrained_WE \
                            --num_topics $NUM_TOPICS \
                            --dataset $DATASET \
                            --seed $seed \
                            --wandb_on \
                            --wandb_prj $WANDB_PROJECT \
                            --wandb_name "${MODEL_NAME}_${DATASET}_top${NUM_TOPICS}_weight_loss_ECR${weight_loss_ECR}_seed${seed}" \
                            --verbose
    done
done