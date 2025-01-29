import torch
import models_vit as vits
import models_mae as maes
from util.pos_embed import interpolate_pos_embed
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from typing import Iterable



#variables 

models = ['tiny', 'small', 'base']
models = [f'/media/gabriel/BA1041B410417881/Users/gabrielc/Projects/mae/others/output_epochs/pre_train/{x}' for x in models]

dataset = '/home/gabriel/Downloads/castas-huge2-split'
input_size = 224


def get_model(weights_path, arch, mode='mae', global_pool=True, device='cpu'):
    
    model = vits.__dict__[arch](
        num_classes=0,
        drop_path_rate=0,
        global_pool=global_pool,
        )
    
    if mode == 'mae':
        checkpoint = torch.load(weights_path, map_location=device)
        print("Load pre-trained DECODER checkpoint from: %s" % weights_path)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        print(model.head)
       
        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        print(model.head)
        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)


    else:
        state = torch.load(weights_path, map_location='cpu')
        print(state.keys())
        msg = model.load_state_dict(state['model'], strict=False)
    
    print(msg)
        
    for p in model.parameters():
        p.requires_grad = False

    model.eval()
    model.to(device)

    return model

#method that train a k-nn classifier based on the features extracted from the model
def train_knn_classifier(model, data_loader: Iterable, device: torch.device, k: int = 5):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for (samples, targets) in data_loader:
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            #Extract features from the model
            feature = model.forward_representations(samples)
            features.append(feature.cpu().numpy())
            labels.append(targets.cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Train k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(features, labels)

    return knn

def evaluate_knn_classifier(knn, model, data_loader: Iterable, device: torch.device):
    model.eval()
    features = []
    true_labels = []

    with torch.no_grad():
        for (samples, targets) in data_loader:
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Extract features from the model
            feature = model.forward_representations(samples)
            features.append(feature.cpu().numpy())
            true_labels.append(targets.cpu().numpy())

    features = np.concatenate(features, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    # Predict using k-NN classifier
    pred_labels = knn.predict(features)
    f1 = f1_score(true_labels, pred_labels, average='macro')
    accuracy = accuracy_score(true_labels, pred_labels)

    return f1, accuracy


transform_eval = transforms.Compose([
            transforms.Resize(input_size, interpolation=3),  # 3 is bicubic
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

dataset_eval_train = datasets.ImageFolder(os.path.join(dataset, 'train'), transform=transform_eval)
dataset_eval_validation = datasets.ImageFolder(os.path.join(dataset, 'val'), transform=transform_eval)







