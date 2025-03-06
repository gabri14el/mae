import torch
import models_mae
import torchvision.transforms as transforms
from util.datasets import GaussianBlur
import torchvision.datasets as datasets
import os
import numpy as np
import json
import tqdm
import umap


#biggest confusions
selected_varieties = ['EC', 'MC', 'MR', 'SM', 'TF', 'DB', 'MF', 'FG', 'DT']


experiments = ['/media/gabriel/7739-DDF5/Gabriel/Projects/maebt/vits/pretrain/coeffv2/0-1/']
epochs = [0, 99, 199, 399, 499, 699, 899, 999]
architectures = 'maebt_vit_small_patch16'
norm_pix_loss = 0.0
projector = '1024-1024-1024'
input_size = 224
dataset_path = '/home/gabriel/Downloads/castas-huge2-split'
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
    data_loader_train = torch.utils.data.DataLoader(ds, sampler=sampler, batch_size=48)

    ds_val = datasets.ImageFolder(os.path.join(dataset, 'val'), transform=Transform(bt_strong_augmentation, input_size))
    sampler = torch.utils.data.SequentialSampler(ds)
    data_loader_val = torch.utils.data.DataLoader(ds, sampler=sampler, batch_size=48)

    return data_loader_train, data_loader_val

def extract_bt_details_trained_models(path_to_model,data_loader_train, architecture, projector, bt_strong_augmentation, input_size, bt_coef, dataset_path, bt_mode, epoch, checkpoint='checkpoint-{}.pth'):
    print('read the model')
    model = read_model(os.path.join(path_to_model, checkpoint.format(epoch)), architecture, projector, input_size)

    model.to('cpu')
    model.eval()

    print('model in cpu')

    projector_dim = int(projector.split('-')[-1])
    accum_matrix = np.zeros((projector_dim, projector_dim))
    accum_on = 0
    accum_off = 0

    print('created accum_matrix')

    count = 0
    tqdm_loader = tqdm.tqdm(enumerate(data_loader_train), total=len(data_loader_train))
    for data_iter_step, ((x1, x2), _) in tqdm_loader:
        #print('iteration: ', data_iter_step)
        count += 1
        x1 = x1.to('cpu')
        x2 = x2.to('cpu')

        mae_loss1, mae_loss2, bt_loss,c, bt_mixup_loss, on, off, pred1, pred2, mask1, mask2, latent1, latent2 = model(x1, x2, mask_ratio=0.6, bt_coef=bt_coef, bt_mode=bt_mode)


        accum_matrix = np.add(accum_matrix, c.detach().numpy())
        accum_on = accum_on + on.detach().numpy()
        accum_off = accum_off + off.detach().numpy()
        tqdm_loader.update()

    
    print(data_iter_step+1, count, len(data_loader_train))
    accum_matrix = accum_matrix / (len(data_loader_train))
    accum_on = accum_on / (len(data_loader_train))
    accum_off = accum_off / (len(data_loader_train))

    np.savetxt(os.path.join(path_to_model, f'accum_matrix-{epoch}'), accum_matrix, delimiter=',')

    results = {'epoch': epoch, 'train_bt_on_diag': accum_on, 'train_bt_off_diag': accum_off}
    #open txt log to save accum_on and accum_off
    with open(os.path.join(path_to_model, f'log_bt.txt'), 'a') as f:
        f.write(json.dumps(results)+'\n')
    
    del model
    


print('created the datasets')
data_loader_train, data_loader_test = create_datasets(dataset_path, False, input_size)

for i, model in enumerate(experiments):
    for epoch in epochs:
        extract_bt_details_trained_models(model,data_loader_train, architectures, projector, False, input_size, bt_coef, dataset_path, bt_mode, epoch)
        print(f'Finished model {i+1}/{len(experiments)} epoch {epoch}/{epochs[-1]}')

