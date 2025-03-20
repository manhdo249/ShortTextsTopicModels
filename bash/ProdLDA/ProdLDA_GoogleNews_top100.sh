export WANDB_API_KEY="c404830b3fe76c9bae6be1dc53effe3226b28175"

MODEL_NAME=ProdLDA
NUM_TOPICS=100
DATASET=GoogleNewsCluster
GLOBAL_DIR=umap_globalcluster150

WANDB_PROJECT=ProdLDA_100topics_GoogleNews

for dropout in 0.0 0.1 0.2
do
    for seed in $(seq 0 10)
    do
        for en_units in 100 75 50 25
        do
            for lr in 0.002 0.001 0.0005
            do
                CUDA_VISIBLE_DEVICES=0 python main.py  \
                                    --model $MODEL_NAME \
                                    --dropout $dropout \
                                    --num_topics $NUM_TOPICS \
                                    --dataset $DATASET \
                                    --seed $seed \
                                    --en_units $en_units \
                                    --lr $lr \
                                    --wandb_on \
                                    --wandb_prj $WANDB_PROJECT \
                                    --wandb_name "${MODEL_NAME}_${DATASET}_top${NUM_TOPICS}_en_units${en_units}_dropout${dropout}_lr${lr}_seed${seed}" \
                                    --verbose 
            done
        done
    done
done