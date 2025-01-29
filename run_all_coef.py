import os
import sys

bt_coef = [0.1, 0.5, 2, 3, 5, 7, 10]
bt_coef = sorted(bt_coef, key=lambda x: float(x), reverse=True)

model = 'small'

os.chdir('/home/gabriel/Projects/mae')
sys.path.append('/home/gabriel/Projects/mae')

for i, c in enumerate(bt_coef):
    '''
    command = f"""/home/gabriel/anaconda3/envs/pytorch/bin/python /home/gabriel/Projects/mae/main_pretrain_merged.py --batch_size 64 \
    --epochs 200 \
    --model maebt_vit_small_patch16 \
    --mask_ratio 0.60 \
    --warmup_epochs 10 \
    --data_path /media/gabriel/BA1041B410417881/Users/gabrielc/Datasets/all_plus_utad/ \
    --blr 1e-3 --weight_decay 0.05 \
    --num_workers 1 \
    --world_size 1 \
    --output /media/gabriel/7739-DDF5/Gabriel/Projects/maebt/vits/pretrain/coeff/{str(c).replace('.', '-')} \
    --log /media/gabriel/7739-DDF5/Gabriel/Projects/maebt/vits/pretrain/coeff/{str(c).replace('.', '-')} \
    --knn_eval \
    --knn_dataset /home/gabriel/Downloads/castas-huge2-split \
    --bt_loss_coef {c * 1e-3}"""

    '''

    command = f"""/home/gabriel/anaconda3/envs/pytorch/bin/python main_finetune.py \
            --batch_size 192 \
            --model vit_{model}_patch16  \
            --finetune /media/gabriel/7739-DDF5/Gabriel/Projects/maebt/vits/pretrain/coeff/{str(c).replace('.', '-')}/checkpoint-199.pth \
            --epochs 100 \
            --blr 1e-3 --layer_decay 0.65 \
            --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
            --warmup_epochs 10 --dist_eval --data_path /home/gabriel/Downloads/castas-huge2-split \
            --output_dir /media/gabriel/7739-DDF5/Gabriel/Projects/maebt/vits/finetune/coeff/{str(c).replace('.', '-')} \
            --nb_classes 43 --num_workers 1 --log /media/gabriel/7739-DDF5/Gabriel/Projects/maebt/vits/finetune/coeff/{str(c).replace('.', '-')} \
            --extra_test_dataset /media/gabriel/7739-DDF5/Gabriel/Datasets/2024/processed/compilado/ --experiment_name "coef_study" \
            --run_name {c}_epoch{199}"""

    print(command)
    #os.system(command)

    command = f"""/home/gabriel/anaconda3/envs/pytorch/bin/python main_finetune.py \
            --batch_size 192 \
            --model vit_{model}_patch16  \
            --finetune /media/gabriel/7739-DDF5/Gabriel/Projects/maebt/vits/pretrain/coeff/{str(c).replace('.', '-')}/checkpoint-99.pth \
            --epochs 100 \
            --blr 1e-3 --layer_decay 0.65 \
            --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
            --warmup_epochs 10 --dist_eval --data_path /home/gabriel/Downloads/castas-huge2-split \
            --output_dir /media/gabriel/7739-DDF5/Gabriel/Projects/maebt/vits/finetune/coeff/{str(c).replace('.', '-')} \
            --nb_classes 43 --num_workers 1 --log /media/gabriel/7739-DDF5/Gabriel/Projects/maebt/vits/finetune/coeff/{str(c).replace('.', '-')} \
            --extra_test_dataset /media/gabriel/7739-DDF5/Gabriel/Datasets/2024/processed/compilado/ --experiment_name "coef_study" \
            --run_name {c}_epoch{99}"""

    print(command)
    #os.system(command)