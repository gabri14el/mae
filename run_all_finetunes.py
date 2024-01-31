import os
import sys

models = ['base', 'small', 'tiny']
#ratios = [50, 60, 75]
ratios = [.60]
batches = [80, 128, 160]


main_dir = os.path.join('output', 'pre_train')
os.chdir('/home/gabriel/Projects/mae')
sys.path.append('/home/gabriel/Projects/mae')

for m, model in enumerate(models):
    for ratio in ratios:
        model_dir = os.path.join(main_dir, model, str(ratio))
        command = f"""/home/gabriel/anaconda3/envs/pytorch/bin/python main_finetune.py \
            --batch_size 80 \
            --model vit_{model}_patch16  \
            --finetune /home/gabriel/Projects/mae/output/pre_train/{model}_{ratio}/checkpoint-499.pth \
            --epochs 100 \
            --blr 1e-3 --layer_decay 0.65 \
            --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
            --warmup_epochs 10 --dist_eval --data_path /home/gabriel/Downloads/castas-huge2-split \
            --output_dir ./output/ft/{model}-{ratio} \
            --nb_classes 43 --num_workers 1 --log ./output/ft/{model}-{ratio}"""
        
        command = f"""python main_pretrain.py --batch_size {batches[m]} \
        --epochs 3000 \
        --model mae_vit_{model}_patch16 \
        --mask_ratio {ratio} \
        --warmup_epochs 10 \
        --data_path /media/gabriel/BA1041B410417881/Users/gabrielc/Datasets/all_plus_utad \
        --blr 1.5e-4 --weight_decay 0.05 \
        --num_workers 1 \
        --world_size 1 \
        --output output_epochs/pretrain/{model} \
        --log output_epochs/pretrain/{model} \
        --start_epoch 0"""

        print(command)
        os.system(command)
