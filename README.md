# -DP00AY81_3001_DRDetection
This is for the coursework of DP00AY81-3001 Advancing AI-Driven Oculomics for Eye and Systemic Diseases

Please download/prepare the pretrained weights first and then run the slurm script provided.

Commands like: 

```
torchrun --nproc_per_node=1 main_finetune.py \
    --model DinoV2_Vanliia \
    --cls_task 2 \
    --nb_classes 2 \
    --savemodel \
    --global_pool \
    --batch_size 32 \
    --world_size 1 \
    --epochs 100 \
    --lr 0.0002 \
    --lrd_total 0.1 \
    --layer_decay 1 \
    --weight_decay 0.05 --drop_path 0.2 \
    --data_path APTOS2019 \
    --input_size 224 \
    --task vanlliaDinoV2_224_2_class_TVERSKY \
    --finetune None \
    --loss_func TVERSKY
wait
