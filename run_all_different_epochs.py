import os
import sys

#models = ['base', 'small', 'tiny']
models = ['base']
models = ['alternative_bt_loss-0-1-augmentation']
#ratios = [50, 60, 75]
ratios = [.60]
epochs = ['0', '99', '199', '500', '1000', '1500', '2000', '2500', '2999']
epochs = [0, 99, 199, 399, 499, 699, 899, 999]
#epochs = [str(x) for x in range(0, 3000, 250)]
#epochs = ['0', '99', '199', '499']
epochs = sorted(epochs, key=lambda x: int(x), reverse=True)
epochs = [str(x) for x in epochs]
#batches = [80, 128, 160]
batches = [192]

experiment_name = 'bt_mae_alternative-bt_augmentation'

ratio = 0.6


#main_dir = os.path.join('output_epochs', 'pre_train')
main_dir = '/media/gabriel/7739-DDF5/Gabriel/Projects/maebt/vits/'
os.chdir('/media/gabriel/7739-DDF5/Gabriel/Projects/maebt/vits/')
sys.path.append('/home/gabriel/Projects/mae')

for m, model in enumerate(models):
    for epoch in epochs:
        #model_dir = os.path.join(main_dir, 'pre-train', model, f'checkpoint-{epoch}.pth')
        #log_dir = os.path.join(main_dir, 'finetune', model, epoch)

        model_dir = os.path.join(main_dir, 'pretrain', model, f'checkpoint-{epoch}.pth')
        log_dir = os.path.join(main_dir, 'finetune', model, epoch)
        
        '''
        command = f"""/home/gabriel/anaconda3/envs/pytorch/bin/python main_finetune.py \
            --batch_size {batches[m]} \
            --model vit_{model}_patch16  \
            --experiment_name bigger_dataset \
            --finetune {model_dir} \
            --epochs 100 \
            --blr 1e-3 --layer_decay 0.65 \
            --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
            --warmup_epochs 10 --dist_eval --data_path /home/gabriel/Downloads/castas-huge2-split \
            --output_dir {log_dir} \
            --nb_classes 43 --num_workers 1 --log {log_dir}"""

        '''
        command = f"""/home/gabriel/anaconda3/envs/pytorch/bin/python /home/gabriel/Projects/mae/main_finetune.py \
            --batch_size {batches[m]} \
            --model vit_{model}_patch16  \
            --experiment_name {experiment_name+'FT'} \
            --finetune {model_dir} \
            --epochs 100 \
            --blr 1e-3 --layer_decay 0.65 \
            --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
            --warmup_epochs 10 --dist_eval --data_path /home/gabriel/Downloads/castas-huge2-split \
            --output_dir {log_dir} \
            --nb_classes 43 --num_workers 1 --log {log_dir} \
            --solo \
            --extra_test_dataset /media/gabriel/7739-DDF5/Gabriel/Datasets/2024/processed/compilado/ \
            --run_name {model}_{epoch}"""

        command = f"""/home/gabriel/anaconda3/envs/pytorch/bin/python /home/gabriel/Projects/mae/main_finetune.py \
            --batch_size {batches[m]} \
            --model vit_small_patch16  \
            --experiment_name {experiment_name+'FT'} \
            --finetune {model_dir} \
            --epochs 100 \
            --blr 1e-3 --layer_decay 0.65 \
            --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
            --warmup_epochs 10 --dist_eval --data_path /home/gabriel/Downloads/castas-huge2-split \
            --output_dir {log_dir} \
            --nb_classes 43 --num_workers 1 --log {log_dir} \
            --extra_test_dataset /media/gabriel/7739-DDF5/Gabriel/Datasets/2024/processed/compilado/ \
            --run_name {model}_{epoch}"""

        print(command)
        os.system(command)

        log_dir = os.path.join(main_dir, 'linearprobing', model, epoch)

        command = f"""/home/gabriel/anaconda3/envs/pytorch/bin/python /home/gabriel/Projects/mae/main_finetune.py \
            --batch_size {batches[m]} \
            --model vit_{model}_patch16  \
            --experiment_name {experiment_name+'LP'} \
            --finetune {model_dir} \
            --epochs 100 \
            --blr 1e-3 --layer_decay 0.65 \
            --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
            --warmup_epochs 10 --dist_eval --data_path /home/gabriel/Downloads/castas-huge2-split \
            --output_dir {log_dir} \
            --nb_classes 43 --num_workers 1 --log {log_dir} --run_name {model}_{epoch} \
            --solo \
            --extra_test_dataset /media/gabriel/7739-DDF5/Gabriel/Datasets/2024/processed/compilado/ \
            --freeze_backbone"""
        
        command = f"""/home/gabriel/anaconda3/envs/pytorch/bin/python /home/gabriel/Projects/mae/main_finetune.py \
            --batch_size {batches[m]*4} \
            --model vit_small_patch16  \
            --experiment_name {experiment_name+'LP'} \
            --finetune {model_dir} \
            --epochs 100 \
            --blr 1e-3 --layer_decay 0.65 \
            --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
            --warmup_epochs 10 --dist_eval --data_path /home/gabriel/Downloads/castas-huge2-split \
            --output_dir {log_dir} \
            --nb_classes 43 --num_workers 1 --log {log_dir} --run_name {model}_{epoch} \
            --extra_test_dataset /media/gabriel/7739-DDF5/Gabriel/Datasets/2024/processed/compilado/ \
            --freeze_backbone"""

        #print(command)
        #os.system(command)