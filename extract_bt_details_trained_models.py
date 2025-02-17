import torch
import models_mae
import torchvision.transforms as transforms
from util.datasets import GaussianBlur
import torchvision.datasets as datasets
import os
import numpy as np
import json


experiments = ['/media/gabriel/7739-DDF5/Gabriel/Projects/maebt/vits/pretrain/coeffv2/0-1/']
epochs = [0, 99, 199, 399, 499, 699, 899, 999]
architectures = 'maebt_vit_small_patch16'
norm_pix_loss = 0.0
projector = '1024-1024-1024'
input_size = 224
dataset_path = '/media/gabriel/BA1041B410417881/Users/gabrielc/Datasets/all_plus_utad/'
bt_coef = 0.0001
bt_mode = 'default'


class Transform:
    def __init__(self, bt_strong_augmentation, input_size):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        if bt_strong_augmentation:
            self.transform_prime = transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )
        
        else:
            self.transform_prime = transforms.Compose([
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),
                #transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.7),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


def read_model(path_to_model, architecture, projector, input_size, model=None):
    
    if model is None:
        model = models_mae.__dict__[architecture](norm_pix_loss=False, 
                                                        barlowtwins_projetcor=projector,
                                                        img_size=input_size)

    model.load_state_dict(torch.load(path_to_model)['model'])

    return model

def create_datasets(dataset, bt_strong_augmentation, input_size):
    ds = datasets.ImageFolder(os.path.join(dataset, 'train'), transform=Transform(bt_strong_augmentation, input_size))
    sampler = torch.utils.data.SequentialSampler(ds)
    data_loader_train = torch.utils.data.DataLoader(ds, sampler=sampler, batch_size=12, num_workers=1, pin_memory=True)

    return data_loader_train

def extract_bt_details_trained_models(path_to_model, architecture, projector, bt_strong_augmentation, input_size, bt_coef, dataset_path, bt_mode, epoch, checkpoint='checkpoint-{}.pth'):
    model = read_model(os.path.join(path_to_model, checkpoint.format(epoch)), architecture, projector, input_size)
    data_loader_train = create_datasets(dataset_path, bt_strong_augmentation, input_size)

    model.to('cpu')
    model.eval()

    projector_dim = int(projector.split('-')[-1])
    accum_matrix = torch.tensor(np.zeros((projector_dim, projector_dim))).to('cpu')
    accum_on = torch.tensor(0).to('cpu')
    accum_off = torch.tensor(0).to('cpu')

    count = 0
    for data_iter_step, ((x1, x2), _) in enumerate(data_loader_train):
        count += 1
        x1 = x1.to('cpu')
        x2 = x2.to('cpu')

        mae_loss1, mae_loss2, bt_loss, c, on, off, pred1, pred2, mask1, mask2, latent1, latent2 = model(x1, x2, mask_ratio=0.6, bt_coef=bt_coef, bt_mode=bt_mode)

        accum_matrix = torch.add(accum_matrix, c)
        accum_on = torch.add(accum_on, on)
        accum_off = torch.add(accum_off, off)

    
    print(data_iter_step+1, count)
    accum_matrix = accum_matrix.detach().cpu().numpy() / (count)
    accum_on = accum_on.detach().cpu().numpy() / (count)
    accum_off = accum_off.detach().cpu().numpy() / (count)

    np.savetxt(os.path.join(path_to_model, f'accum_matrix-{epoch}'), accum_matrix, delimiter=',')

    results = {'epoch': epoch, 'train_bt_on_diag': accum_on, 'train_bt_off_diag': accum_off}
    #open txt log to save accum_on and accum_off
    with open(os.path.join(path_to_model, f'log_bt.txt'), 'a') as f:
        f.write(json.dumps(results)+'\n')

for i, model in enumerate(experiments):
    for epoch in epochs:
        extract_bt_details_trained_models(model, architectures, projector, False, input_size, bt_coef, dataset_path, bt_mode, epoch)
        print(f'Finished model {i+1}/{len(experiments)} epoch {epoch}/{epochs[-1]}')

