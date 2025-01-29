# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib
from sklearn import metrics
import itertools
import numpy as np
import math
import torch
import seaborn as sns
import pandas as pd

MAX = 65535

def step_decay(epoch, initial_lrate=0.01):
    flattern_factor = initial_lrate ** 2.25
    epochs_drop = 5.0
    #drop modelado como modelado no artigo
    drop = initial_lrate **(flattern_factor/epochs_drop)
    
    lrate = initial_lrate * math.pow(drop,  
            math.floor((epoch)/epochs_drop))
    return lrate

def normalize_rgb_ln(X, preprocess=None):
    a = np.log(X)/np.log(65535.0)
    a = a * 255
    if not preprocess:
        return a.astype('uint8')
    return preprocess(a.astype('uint8'))

def plot_confusion_matrix_sns(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Greens, 
                        only_heatmap=False):
    
    min = np.min(cm)
    max = np.max(cm)
    #print(cm)
    if normalize:
        #print(cm.sum(axis=1)[:, np.newaxis])
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]+0.0000001
        cm = np.around(cm, decimals=4)
        min = 0
        max = 1
    
    #print(cm)
    df_cm = pd.DataFrame(cm, index = classes, columns=classes)
    
    
    off_diag_mask = np.eye(*cm.shape, dtype=bool)
    if(len(classes) < 20):
      figsize = len(classes)
    elif only_heatmap:
      figsize = 3
    else:
       figsize = 0.65*len(classes)

    fig = plt.figure(figsize=(figsize+1, figsize), dpi=300)

    if only_heatmap:
      sns.heatmap(df_cm, annot=False,cmap=cmap, vmin=min, vmax=max)
    else:
      sns.heatmap(df_cm, annot=True,cmap=cmap, vmin=min, vmax=max)

    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


    return fig
        

#DEFINITION OF TEST METHODS
def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Greens,
                        only_heatmap=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """


    if(len(classes) < 20):
      figsize = len(classes)
    elif only_heatmap:
      figsize = 3
    else:
       figsize = 0.65*len(classes)

    fig = plt.figure(figsize=(figsize+1, figsize), dpi=300)

    min = cm.min()
    max = cm.max()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=4)
        min = 0
        max = 1

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.clim(min, max)

    if not only_heatmap:
      tick_marks = np.arange(len(classes))
      plt.xticks(tick_marks, classes, rotation=45)
      plt.yticks(tick_marks, classes)
    
    
    if not only_heatmap:
      thresh = cm.max() / (2/3.)
      for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
          plt.text(j, i, cm[i, j],
              horizontalalignment="center",
              color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig

def confusion_matrix(test_data_generator, model, return_fig=False, class_labels=None, steps=None, mode='tensorflow', sns=False, normalize=False, return_images_paths=False, only_heatmap=False, cmap=plt.cm.Greens):
  
  #tensorflow mode
  #test_data_generator.reset()
  if mode == 'tensorflow':
    if steps == None:
        steps=test_data_generator.samples
    predictions = model.predict(test_data_generator, steps=steps)
    #print(predictions)
    # Get most likely class
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = list(test_data_generator.labels)
  
  #pytorch mode 
  else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    true_classes = []
    predicted_classes = []
    with torch.no_grad():
        for images, labels in test_data_generator: 
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            true_classes.extend(labels.cpu().numpy().tolist())
            predicted_classes.extend(predicted.cpu().numpy().tolist())

    equal = np.equal(true_classes, predicted_classes)
    images_paths = list(zip(true_classes, equal, test_data_generator.dataset.samples))
  
  if class_labels == None:
    class_labels = [str(x) for x in np.unique(true_classes)]
  #print(class_labels)  
  #print(len(true_classes))
  report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels, digits=4)
  cm = metrics.confusion_matrix(true_classes, predicted_classes)
  print(report)
  if not sns:
    fig = plot_confusion_matrix(cm, class_labels, normalize=normalize, only_heatmap=only_heatmap, cmap=cmap)
  else:
    fig = plot_confusion_matrix_sns(cm, class_labels, normalize=normalize, only_heatmap=only_heatmap, cmap=cmap)
  if return_fig and return_images_paths:
    return report, fig, images_paths
  if return_fig:
    return report, fig
  if return_images_paths:
    return report, images_paths
  
  return report