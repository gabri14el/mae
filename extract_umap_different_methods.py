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
import umap.plot
import models_vit
import vision_transformer as vits
import matplotlib.pyplot as plt
import seaborn as sns

#biggest confusions
selected_varieties = ['EC', 'MC', 'MR', 'SM', 'TF', 'DB', 'MF', 'FG', 'DT']


varieties = {'tinto cao': 'TC', 'tinta francisca': 'TF', 'alicante': 'AC', 'alveralhao': 'AV', 'arinto': 'AT', 'bastardo': 'BT', 'boal': 'BA', 'cabernet franc': 'CF', 'cabernet sauvignon': 'CS', 'carignon noir': 'CN', 'cercial': 'CC', 'chardonnay': 'CD', 'codega': 'CG', 'codega do larinho': 'CR', 'cornifesto': 'CT', 'donzelinho': 'DZ', 'donzelinho branco': 'DB', 'donzelinho tinto': 'DT', 'esgana cao': 'EC', 'fernao pires': 'FP', 'folgasao': 'FG', 'gamay': 'GM', 'gouveio': 'GV', 'malvasia corada': 'MC', 'malvasia fina': 'MF', 'malvasia preta': 'MP', 'malvasia rei': 'MR', 'merlot': 'ML', 'moscatel galego': 'MG', 'moscatel galego roxo': 'MX', 'mourisco tinto': 'MT', 'pinot blanc': 'PB', 'rabigato': 'RB', 'rufete': 'RF', 'samarrinho': 'SM', 'sauvignon blanc': 'SB', 'sousao': 'SS', 'tinta amarela': 'TA', 'tinta barroca': 'TB', 'tinta femea': 'TM', 'tinta roriz': 'TR', 'touriga francesa': 'TS', 'touriga nacional': 'TN', 'viosinho': 'VO'}
selected_varieties = {x:y for x,y in varieties.items() if y in selected_varieties}

experiments = ['/media/gabriel/7739-DDF5/Gabriel/Projects/maebt/vits/pretrain/alternative_bt_loss-0-1-augmentation/']
mode = ['dino']
epochs = [0, 99, 199, 399, 499, 699, 899, 999]
#epochs = [0, 99, 199, 500, 1000, 1500, 2000, 2500, 2999]
epochs.sort(reverse=True)
architectures = 'vit_small_patch16'
norm_pix_loss = 0.0
projector = '1024-1024-1024'
input_size = 224
dataset_path = '/home/gabriel/Downloads/castas-huge2-split'
bt_coef = 0.0001
bt_mode = 'default'
from util.pos_embed import interpolate_pos_embed

transform = transforms.Compose([
                transforms.Resize(input_size, interpolation=3),  # 3 is bicubic
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
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


def read_model(path_to_model, architecture, mode='mae'):
    
    model = models_vit.__dict__[architecture](
        num_classes=0,
        drop_path_rate=.0,
        global_pool=True,
    )

    checkpoint = torch.load(path_to_model, map_location='cpu')
    print("Load pre-trained checkpoint from: %s" % path_to_model)
    if  mode == 'mae' or mode == 'maebt':
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        
        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)
        msg = model.load_state_dict(checkpoint_model, strict=False)
    
    elif mode == 'solo':
        _new_dict = {}
        state_dict = checkpoint['state_dict']
        for k in model.state_dict().keys():
            if k.startswith('head.') or k.startswith('fc_norm.'):
                print('Skipping: ', k)
            elif 'backbone.'+k in state_dict:
                _new_dict[k] = state_dict['backbone.'+k]
            else:
                print('Not found: ', k)
        checkpoint_model = _new_dict
        msg = model.load_state_dict(checkpoint_model, strict=False)
    elif mode == 'dino':
        model = vits.__dict__[architecture](patch_size=16, num_classes=0)
        embed_dim = model.embed_dim * (4 + int(False))
        checkpoint_key = 'teacher'
        state_dict = checkpoint[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        #print('Pretrained weights found at {} and loaded with msg: {}'.format(path_to_model, msg))
    elif mode == 'bt':
        print(checkpoint.keys())
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()

        #rename the keys to match the model
        new_checkpoint = {}

        for key, value in state_dict.items():
            if 'module.backbone.'+key in checkpoint_model:
                new_checkpoint[key] = checkpoint_model['module.backbone.'+key]

        checkpoint_model = new_checkpoint
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        print(model.head)
        msg = model.load_state_dict(checkpoint_model, strict=False)
    

    print('model loded with the message', msg)
    return model

def create_datasets(dataset, bt_strong_augmentation, input_size):
    ds = datasets.ImageFolder(os.path.join(dataset, 'train'), transform=transform)
    sampler = torch.utils.data.SequentialSampler(ds)
    data_loader_train = torch.utils.data.DataLoader(ds, sampler=sampler, batch_size=128, num_workers=1, pin_memory=True)

    ds_val = datasets.ImageFolder(os.path.join(dataset, 'val'), transform=transform)
    sampler = torch.utils.data.SequentialSampler(ds)
    data_loader_val = torch.utils.data.DataLoader(ds, sampler=sampler, batch_size=128, num_workers=1, pin_memory=True)

    classes = list(ds.class_to_idx.keys())

    return data_loader_train, data_loader_val, classes

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
        accum_on = np.add(accum_on, on.detach().numpy())
        accum_off = np.add(accum_off, off.detach().numpy())
        tqdm_loader.update()

    
    print(data_iter_step+1, count)
    accum_matrix = accum_matrix / (count)
    accum_on = accum_on / (count)
    accum_off = accum_off / (count)

    np.savetxt(os.path.join(path_to_model, f'accum_matrix-{epoch}'), accum_matrix, delimiter=',')

    results = {'epoch': epoch, 'train_bt_on_diag': accum_on, 'train_bt_off_diag': accum_off}
    #open txt log to save accum_on and accum_off
    with open(os.path.join(path_to_model, f'log_bt.txt'), 'a') as f:
        f.write(json.dumps(results)+'\n')
    
    del model

def extract_fetures(model, data_loader, mode='mae'):
    model.to('cpu')
    model.eval()

    _features = []
    _labels = []

    tqdm_loader = tqdm.tqdm(enumerate(data_loader_train), total=len(data_loader_train))
    for data_iter_step, (image, label) in tqdm_loader:
        #print('iteration: ', data_iter_step)

        image, label = image.to('cpu'), label.to('cpu')
        if mode == 'mae' or mode == 'bt':
            features = model.forward_features(image)
        elif mode == 'dino':
            intermediate_output = model.get_intermediate_layers(image, 4)
            features = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)

        _features.append(features.detach().numpy())
        _labels.append(label.detach().numpy())
    
    return np.concatenate(_features, axis=0), np.concatenate(_labels, axis=0)

def umap_calculation(features, labels, filter=None, names=None):
    reducer = umap.UMAP(random_state=35)
    embedding = reducer.fit_transform(features)

    plt.figure(figsize=(17, 10))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=10, alpha=0.7)

    legend_labels = np.unique(labels)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(label)), markersize=8) for label in legend_labels]
    
    if names is None:
        legend_labels = np.unique(labels)
    else:
        legend_labels = names
    plt.legend(handles, legend_labels, title="Classes", loc="upper left", fontsize='small', bbox_to_anchor=(1, 1))

    plt.title("UMAP Projection with Labels")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    #plt.show()
    #val_image = umap.plot.points(reducer, labels=labels)
    #val_image.figure.set_size_inches(10, 10, forward=True)
    #plt.legend(title="Class")
    val_image = plt.gcf()
    
    if filter is not None:
        mask = np.isin(labels, filter)
        # Filter the embedding and labels
        embedding_filtered = embedding[mask]
        y_filtered = labels[mask]
        legend_labels = np.unique(y_filtered)

        # Plot
        plt.figure(figsize=(17, 10))
        scatter = plt.scatter(embedding_filtered[:, 0], embedding_filtered[:, 1], c=y_filtered, cmap='Spectral', s=10, alpha=0.7)
        #sns.scatterplot(x=embedding_filtered[:, 0], y=embedding_filtered[:, 1], hue=y_filtered, legend='full')
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(label)), markersize=8) for label in legend_labels]
        
        if names is None:
            legend_labels = np.unique(y_filtered)
        else:
            legend_labels = [names[f] for f in filter]
        plt.legend(handles, legend_labels, title="Classes", loc="upper left", fontsize='small', bbox_to_anchor=(1, 1))


        plt.title("UMAP Projection with Labels")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        fig = plt.gcf()
        #fig.set_size_inches(13, 10, forward=True)

        return val_image, fig

    return val_image


    


print('created the datasets')
data_loader_train, data_loader_test, classes = create_datasets(dataset_path, False, input_size)

print(classes)
filter = [classes.index(x.title()) for x in list(selected_varieties.keys())]

print('filter for the classes:', selected_varieties.keys())
for i, model_path in enumerate(experiments):
    for epoch in epochs:
        model = read_model(os.path.join(model_path, 'checkpoint-{}.pth'.format(epoch)), architectures)
        #train_features, train_labels = extract_fetures(model, data_loader_train)
        val_features, val_labels = extract_fetures(model, data_loader_test, mode='mae')
        val_image, fig = umap_calculation(val_features, val_labels, filter=filter, names=[varieties[c.lower()] for c in classes]) #
        fig.savefig(os.path.join(model_path, f'umap_filtred-{epoch}.png'), dpi=300)
        #save val_image
        #val_image.figure.savefig(os.path.join(model_path, f'umap-{epoch}.png'), dpi=300)
        val_image.savefig(os.path.join(model_path, f'umap-{epoch}.png'), dpi=300)
        print(f'Finished model {i+1}/{len(experiments)} epoch {epoch}/{epochs[-1]}')

