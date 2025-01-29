import os
import sys

#models = ['base', 'small', 'tiny']
models = ['base']
#ratios = [50, 60, 75]
ratios = [.60]
#epochs = ['0', '99', '199', '500', '1000', '1500', '2000', '2500', '2999']
epochs = ['0', '99', '199', '499']
epochs = sorted(epochs, key=lambda x: int(x), reverse=True)
#batches = [80, 128, 160]
percentages = [1, 5, 10, 25, 50, 75]
batches = [80]

#maes = ['/home/gabriel/Projects/mae/mae_pretrain_vit_base.pth', '/media/gabriel/BA1041B410417881/Users/gabrielc/Projects/mae/others/output_epochs/pre_train/base/checkpoint-2000.pth']
#aliases = ['imagenet', 'ours']
maes = ['/media/gabriel/BA1041B410417881/Users/gabrielc/Projects/mae/bigger_dataset2/pretrain/base/checkpoint-2500.pth',
        '/home/gabriel/Projects/mae/mae_pretrain_vit_base.pth',
        '/media/gabriel/BA1041B410417881/Users/gabrielc/Projects/mae/others/output_imagenet_v4/pretrain/base/checkpoint-2000.pth',
        '/media/gabriel/BA1041B410417881/Users/gabrielc/Projects/mae/others/output_epochs/pre_train/base/checkpoint-2000.pth']
aliases = ['ourPuls', 'imagenet', 'imagenetOurs', 'ours']
ratio = 0.6


#main_dir = os.path.join('output_epochs', 'pre_train')
main_dir = 'output_MergedDatasets'
os.chdir('/home/gabriel/Projects/mae')
sys.path.append('/home/gabriel/Projects/mae')
dataset_dir = '/home/gabriel/Downloads/castas-huge3-split/'

for m, model in enumerate(maes):
    command = f"""/home/gabriel/anaconda3/envs/pytorch/bin/python main_finetune.py \
        --batch_size 80 \
        --model vit_base_patch16  \
        --experiment_name merged_datasets \
        --finetune {model} \
        --epochs 100 \
        --blr 1e-3 --layer_decay 0.65 \
        --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
        --warmup_epochs 10 --dist_eval --data_path {dataset_dir} \
        --output_dir /home/gabriel/Projects/mae/{main_dir}/ft/{aliases[m]} \
        --nb_classes 43 --num_workers 1 --log /home/gabriel/Projects/mae/{main_dir}/ft/{aliases[m]}"""

    print(command)
    os.system(command)
