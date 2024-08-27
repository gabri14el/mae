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
batches = [80]

ratio = 0.6


#main_dir = os.path.join('output_epochs', 'pre_train')
main_dir = '/home/gabriel/Projects/mae/output_haugmentationv2/'
os.chdir('/home/gabriel/Projects/mae')
sys.path.append('/home/gabriel/Projects/mae')

for m, model in enumerate(models):
    for epoch in epochs:
        model_dir = os.path.join(main_dir, 'pretrain', model, f'checkpoint-{epoch}.pth')
        log_dir = os.path.join(main_dir, 'ft', model, epoch)
        command = f"""/home/gabriel/anaconda3/envs/pytorch/bin/python main_finetune.py \
            --batch_size {batches[m]} \
            --model vit_{model}_patch16  \
            --experiment_name data_augmentation_initialisationv2 \
            --finetune {model_dir} \
            --epochs 100 \
            --blr 1e-3 --layer_decay 0.65 \
            --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
            --warmup_epochs 10 --dist_eval --data_path /home/gabriel/Downloads/castas-huge2-split \
            --output_dir {log_dir} \
            --nb_classes 43 --num_workers 1 --log {log_dir}"""

        print(command)
        os.system(command)
