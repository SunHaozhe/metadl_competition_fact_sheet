#!/usr/bin/env python
# coding: utf-8


#Imports
import sys
import argparse

#=================================================
# Arguments
#=================================================

parser=argparse.ArgumentParser()

parser.add_argument('--CSV_PATH', required=True, help='Path of the CSV file which contains the label and image name')

parser.add_argument('--CSV_WITH_TAB', default=False, help='True if CSV is tab separated otherwise false', action="store_true")

parser.add_argument('--IMAGE_PATH', required=True, help='Path of the directory where images to be used in this experiement are saved')

parser.add_argument('--PREDICTIONS_PATH', type=str, required=True, 
    help="Path of the directory where the results of this experiments will bee saved logs.txt, super_categories.txt and one directory for each super_category which contains categories.txt, categories_auc.txt, logs.txt, some figures etc ")

parser.add_argument('--LABEL_COLUMN', required=True, help='label column name in csv')

parser.add_argument('--IMAGE_COLUMN', required=True, help='image column name in csv')

parser.add_argument('--CATEGORIES_TO_COMBINE', type=int, default=5, 
    help='number of categories to combine to make a super-category or a classification task')

parser.add_argument('--IMAGES_PER_CATEGORY', type=int, default=10, 
    help='number of images per category')

parser.add_argument('--MAX_EPISODES', type=int, default=None, 
    help='maximum limit on episodes/super-categories')

parser.add_argument('--USE_NORMALIZATION', action='store_true', default=False, 
    help='Normalize the input images according to the way neural networks were pretrained on ImageNet')

parser.add_argument('--EMBEDDING', action='store_true', default=False, 
    help="""Select half of the available classes to train the neural network embedding extractor 
    (to avoid using that trained on ImageNet), the remaining classes are used to generate factsheet.""")


args=parser.parse_args()




#=================================================
# Settings
#=================================================

# Path of the CSV file which contains the label and image name
CSV_PATH = args.CSV_PATH

# True if CSV is tab separated otherwise false
CSV_WITH_TAB = args.CSV_WITH_TAB

# Path of the directory where images to be used in this experiement are saved
IMAGE_PATH = args.IMAGE_PATH

# Path of the directory where the results of this experiments will bee saved
# logs.txt, super_categories.txt and one directory for each super_category
# which contains categories.txt, categories_auc.txt, logs.txt, some figures etc 
PREDICTIONS_PATH = args.PREDICTIONS_PATH

# label column name in csv
LABEL_COLUMN = args.LABEL_COLUMN

# image column name in csv
IMAGE_COLUMN = args.IMAGE_COLUMN

# number of categories to combine to make a super-category or a classification task
CATEGORIES_TO_COMBINE = args.CATEGORIES_TO_COMBINE

# number of images per category
# shots = IMAGES_PER_CATEGORY/2
IMAGES_PER_CATEGORY = args.IMAGES_PER_CATEGORY

# maximum limit on episodes/super-categories
# can be none or an integer
MAX_EPISODES = args.MAX_EPISODES

# Normalize the input images according to the way neural networks were pretrained on ImageNet
USE_NORMALIZATION = args.USE_NORMALIZATION

# Select half of the available classes to train the neural network embedding extractor 
# (to avoid using that trained on ImageNet), the remaining classes are used to generate factsheet.
EMBEDDING = args.EMBEDDING


# seed for generating super-categories by the same random combination of categories
SEED = 42





#=================================================
# Imports
#=================================================



import os
import random
import math
import time
import datetime
import copy


import pandas as pd
import numpy as np

import cv2

import gc

import matplotlib.pyplot as plt
import seaborn as sns


params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
plt.rcParams.update(params)



from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader



#=================================================
# Time
#=================================================

t0_start = time.time()
timestamp = str(datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f"))




#=================================================
# Create Prediction Dir
#=================================================

if not os.path.exists(PREDICTIONS_PATH):
    os.makedirs(PREDICTIONS_PATH)


#=================================================
# Setting device
#=================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    

#=================================================
# Load CSV
#=================================================

if CSV_WITH_TAB:
    data = pd.read_csv(CSV_PATH, sep="\t", encoding="utf-8") 
else:
    data = pd.read_csv(CSV_PATH)

print("Data Shape : ", data.shape)


#=================================================
# Prepare new embedding weight
#=================================================

np.random.seed(SEED)

if EMBEDDING:
    from train_embedding_nn import train_embedding_nn
    categories = data[LABEL_COLUMN].unique()
    total_categories = len(categories)
    
    # classes for training the new embedding neural network
    train_categories = np.random.choice(categories, size=total_categories//2, replace=False)
    # classes for the fact sheet results
    categories = np.array([xx for xx in categories if xx not in train_categories])
    total_categories = len(categories)

    print("Training embedding neural network")
    print("{} classes for training embedding neural network, {} classes for the factsheet (AUC).".format(
        len(train_categories), len(categories)))

    train_df = data.loc[data[LABEL_COLUMN].isin(train_categories), [IMAGE_COLUMN, LABEL_COLUMN]]
    
    model_checkpoint_path = train_embedding_nn(train_df, IMAGE_PATH, LABEL_COLUMN, 
        USE_NORMALIZATION, len(train_categories), device, batch_size=64, epochs=20)
else:
    categories = data[LABEL_COLUMN].unique()
    total_categories = len(categories)





#=================================================
# Random Super Categories
#=================================================

random.Random(SEED).shuffle(categories)

iterations_needed = math.ceil(total_categories/CATEGORIES_TO_COMBINE)


print("-----------------------------------------------")
print("Total Categories/Classes : ", total_categories)
print("Iterations required : ", iterations_needed)
print("Categories to combine togather : ", CATEGORIES_TO_COMBINE)



super_categories = np.array_split(categories,iterations_needed)


total_super_categories = len(super_categories)


print("Total Super-Categories : ", total_super_categories)
print("-----------------------------------------------")











#=================================================
# Preparing Super_Categories
#=================================================


super_data = []
for index, super_category in enumerate(super_categories):
    super_dict = {}

    #extracting subset of images per category
    super_category_df = data[data[LABEL_COLUMN].isin(super_category)].groupby(LABEL_COLUMN).sample(n=IMAGES_PER_CATEGORY, random_state=SEED)
    super_category_df['label_cat'] = super_category_df[LABEL_COLUMN].astype('category')
    super_dict['super_category'] = str(index)
    
#     super_dict['categories'] = super_category_df[LABEL_COLUMN].value_counts().index.values
    super_dict['categories'] = super_category_df['label_cat'].cat.categories.values
    
    
    super_dict['images'] = super_category_df[LABEL_COLUMN].value_counts().values
    
    
    train_data, valid_data = train_test_split(
        super_category_df, test_size=0.5, 
        random_state=420, shuffle=True, 
        stratify=super_category_df[LABEL_COLUMN]
    )
    

    
    super_dict['train_labels'] = train_data[LABEL_COLUMN].values
    super_dict['valid_labels'] = valid_data[LABEL_COLUMN].values
    
    super_dict['train_labels_num'] =  train_data['label_cat'].cat.codes.values
    super_dict['valid_labels_num'] = valid_data['label_cat'].cat.codes.values
    
    
  
    
    super_dict['train_data'] = train_data[IMAGE_COLUMN].values
    super_dict['valid_data'] = valid_data[IMAGE_COLUMN].values
    
    super_data.append(super_dict)


    
    
    
    

#=================================================    
# Baseline 
#=================================================






#=================================================
# DATA
#=================================================


class ImgDataset(Dataset):
    def __init__(self, dataset_images, dataset_labels, transform):

        # Transforms
        self.transform = transform
        
        self.images = dataset_images
        self.labels = dataset_labels

        self.data_len = len(self.images)

    def __getitem__(self, index):
        
        
        single_img = self.images[index]
        img_transformed = torch.from_numpy(single_img).long()
        img_transformed = img_transformed.permute(2, 0, 1)
        img_transformed = torch.from_numpy(np.array(img_transformed)).float() / 255.

        
        single_label = self.labels[index]
        single_label = single_label.astype(np.compat.long)
        
        return img_transformed, single_label

    def __len__(self):
        return self.data_len


    
    
    

    

#=================================================
# Make Dataset
#=================================================



def make_dataset(super_data_set, batch_size=64):
    if not USE_NORMALIZATION:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    train_ds=ImgDataset(super_data_set['train_images'],super_data_set['train_labels_num'], transform)
    valid_ds=ImgDataset(super_data_set['valid_images'],super_data_set['valid_labels_num'], transform)
    
    print("############################################")
    print("============================================")
    print("=== Super-Category: ", super_data_set['super_category'])
    print("============================================")
    print("############################################")
    print()
    print("Total Categories: ", len(super_data_set['categories']))
    print("Total Images: ", super_data_set['images'].sum())
    print("Train Data: ", len(train_ds))
    print("Validation Data: ", len(valid_ds))
    print()
    
    data_stats = {
        "total_images" : super_data_set['images'].sum(),
        "train_images" : len(train_ds),
        "valid_images" : len(valid_ds)
    }
    
    
    dataloaders = {
        'val':DataLoader(
            valid_ds, 
            batch_size=batch_size,
            shuffle=False,
        ),
        'train':DataLoader(
            train_ds, 
            batch_size=batch_size,
            shuffle=True,
        )
    }

    dataset_sizes = {
        'val': len(valid_ds),
        'train':len(train_ds)
    }
    
    return dataloaders, dataset_sizes, data_stats





#=================================================
# Model
#=================================================

def getModel(only_train_last_layer=True, number_of_classes=2):
    if not EMBEDDING:
        model = models.resnet18(pretrained=True)
    else:
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(train_categories))
        model.load_state_dict(torch.load(model_checkpoint_path))

    if only_train_last_layer:
        for param in model.parameters():
            param.requires_grad = False
        
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, number_of_classes)
    return model.to(device)











#=================================================
# Training Loop
#=================================================

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=5):
    
    
    print("--------------------------------------------")
    print("Training")
    print("--------------------------------------------")
    
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    
    best_val_acc = 0.0
    best_train_acc = 0.0
    
    loss_history = []
    score_history = []
    
    
    train_loss, train_score, valid_loss, valid_score = [], [], [], []

    print("Epoch: ", end=" ")
    for epoch in range(num_epochs):
        print(epoch, end=" ")
        
        train_predictions, train_ground, valid_predictions, valid_ground = [], [], [], []
        train_predicted_probabilities, valid_predicted_probabilities = [],[]
        

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
#                     inputs = inputs.permute(0, 3, 1, 2)
#                     inputs = torch.from_numpy(np.array(inputs)).float() / 255.
                    
                    outputs = model(inputs)
                    probabilities = F.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                 
                    loss = criterion(outputs, labels)
                    
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # saving prediction and ground truth for future use
                if phase == 'train':
                    train_predictions += list(preds.cpu().detach().numpy())
                    train_ground += list(labels.cpu().detach().numpy())
                    train_predicted_probabilities += list(probabilities.cpu().detach().numpy())
                else:
                    valid_predictions += list(preds.cpu().detach().numpy())
                    valid_ground += list(labels.cpu().detach().numpy())
                    valid_predicted_probabilities += list(probabilities.cpu().detach().numpy())
                
                
            # end dataloader loop

            if phase == 'train':
                scheduler.step()
#                 loss_history.append(running_loss)
#                 score_history.append(running_corrects)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = (running_corrects.double() / dataset_sizes[phase]).item()

#             print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            loss_history, score_history = (train_loss, train_score) if phase == 'train' else (valid_loss, valid_score)
            loss_history.append(epoch_loss)
            score_history.append(epoch_acc)

            
            if phase == 'train' and epoch_acc > best_train_acc:
                 best_train_acc = epoch_acc
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        # end phase loop
        

    time_elapsed = time.time() - since
    print()
    print()
    
    training_time = '{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)
    
    print('Training complete in: {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best train Acc: {:.2f}'.format(best_train_acc))
    print('Best val Acc: {:.2f}'.format(best_val_acc))
    print()

    # load best model weights
    model.load_state_dict(best_model_wts)

    return dict(   
        model = model,
        train_loss = train_loss,
        train_score = train_score,
        train_best_score = round(best_train_acc,2),
        train_ground = train_ground,
        train_predictions = train_predictions,
        train_predicted_probabilities = np.array(train_predicted_probabilities),
        valid_loss = valid_loss,
        valid_score = valid_score,
        valid_best_score = round(best_val_acc,2),
        valid_ground = valid_ground,
        valid_predictions = valid_predictions,
        valid_predicted_probabilities =  np.array(valid_predicted_probabilities),
        training_time = training_time
        
    )






#=================================================
# Plot accuracy and loss
#=================================================

def plot_train_results(train_results, super_category):
    
    
    standard_error = train_results['standard_error']
    y_upper = train_results["valid_score"] + standard_error
    y_lower = train_results["valid_score"] - standard_error
    

    
    print("--------------------------------------------")
    print("Results")
    print("--------------------------------------------")
    
    fig = plt.figure(figsize=(20,8))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(0,len(train_results["train_score"])), train_results["train_score"], label='train')

    plt.plot(range(0,len(train_results["valid_score"])), train_results["valid_score"], label='valid')
    
    
    kwargs = {'color': 'black', 'linewidth': 1, 'linestyle': '--', 'dashes':(5, 5)}
    plt.plot(range(0,len(train_results["valid_score"])), y_lower, **kwargs)
    plt.plot(range(0,len(train_results["valid_score"])), y_upper, **kwargs, label='validation SE (68% CI)')
    
    
    plt.title('Accuracy Plot - ' + super_category, fontsize=20)
    plt.ylabel('Accuracy', fontsize=16)
    plt.xlabel('Training Epochs', fontsize=16)
    plt.ylim(0, 1)
    plt.legend()


    plt.subplot(1, 2, 2)
    plt.plot(range(0,len(train_results["train_loss"])), train_results["train_loss"], label='train')
    plt.plot(range(0,len(train_results["valid_loss"])), train_results["valid_loss"], label='valid')
    

    
    plt.title('Loss Plot - ' + super_category, fontsize=20)
    plt.ylabel('Loss', fontsize=16)
    plt.xlabel('Training Epochs', fontsize=16)
    max_train_loss = max(train_results["train_loss"])
    max_valid_loss = max(train_results["valid_loss"])
    y_max_t_v = max_valid_loss if max_valid_loss > max_train_loss else max_train_loss
    ylim_loss = y_max_t_v if y_max_t_v > 1 else 1
    plt.ylim(0, ylim_loss)
    plt.legend()

  
    plt.show()
    
    super_category_path = os.path.join(PREDICTIONS_PATH,super_category, "train_results.png")
    fig.savefig(super_category_path, dpi=fig.dpi)
    
    
    
def get_error_bar(best_score, valid_examples):
    
    print("--------------------------------------------")
    print("Standard Error")
    print("--------------------------------------------")
    

    
    err = np.sqrt((best_score * (1-best_score))/valid_examples)
    err_rounded_68 = round(err,2)
    err_rounded_95 = round((err_rounded_68 * 2),2)
   
    print('Error (68% CI): +- ' + str(err_rounded_68))
    print('Error (95% CI): +- ' + str(err_rounded_95))
    print()
    return err_rounded_68
    

    
    

#=================================================
# Plot Confusion Matrix
#=================================================

def plot_confusion_matrix(grounds, preds, super_category, categories):
    
    print("--------------------------------------------")
    print("Confusion Matrix")
    print("--------------------------------------------")
    
    num_cat = []
    for ind, cat in enumerate(categories):
        print("Class {0} : {1}".format(ind, cat))
        num_cat.append(ind)
    print()
    
    
    
    cm = confusion_matrix(grounds, preds, labels=num_cat)
    
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_title('Confusion Matrix - '+ super_category, fontsize=20)
    ax.set_xlabel('Predicted labels', fontsize=16)
    ax.set_ylabel('True labels', fontsize=16)
    
    ax.xaxis.set_ticklabels(num_cat)
    ax.yaxis.set_ticklabels(num_cat)
    
    plt.pause(0.1)
    super_category_path = os.path.join(PREDICTIONS_PATH,super_category, "confusion_matrix.png")
    fig.savefig(super_category_path, dpi=fig.dpi)
    

    
    
    
    
    
    
    
    

#=================================================
# Plot Sample Images
#=================================================

def plot_sample_images(super_category, grounds, train_images):
    print("--------------------------------------------")
    print("Sample Images - ", super_category)
    print("--------------------------------------------")

    
    uniques, indexes = np.unique(grounds, return_index=True)
    

    fig = plt.figure(figsize=(20,20))
    k=0
    for i in range(0, len(indexes)):
        fig.add_subplot(1,5,k+1)
        plt.axis('off')
        title = "Label : "+ str(uniques[i])
        plt.title(title,fontsize=16)
        plt.imshow(train_images[uniques[i]])
        k += 1
    plt.pause(0.1)
    print()   
    
    super_category_path = os.path.join(PREDICTIONS_PATH,super_category, "sample_images.png")
    fig.savefig(super_category_path, dpi=fig.dpi, bbox_inches='tight')



    
    
    
    
    

#=================================================    
# Plot Wrongly Classified Images
#=================================================

def plot_wrongly_classified_images(super_category, grounds, preds, valid_images):
    
    print("--------------------------------------------")
    print("Wrongly Classified Images - ", super_category)
    print("--------------------------------------------")
    
    fig = plt.figure(figsize=(20,20))
    k=0
    for i in range(0, len(grounds)):
        if grounds[i] != preds[i]:
            
            fig.add_subplot(1,5,k+1)
            plt.axis('off')
            title = "Orig lbl: "+ str(grounds[i]) + " Pred lbl: " + str(preds[i])
            plt.title(title, fontsize=16)
            plt.imshow(valid_images[i])
            k += 1
        if k == 5:
            break
    plt.pause(0.1)
    print()
    
    super_category_path = os.path.join(PREDICTIONS_PATH,super_category, "wrongly_classified_images.png")
    fig.savefig(super_category_path, dpi=fig.dpi, bbox_inches='tight')
    
def get_wrongly_classified_images_indexes(grounds, preds):
    wrong_images_indexes = []
    for i in range(0, len(grounds)):
        if grounds[i] != preds[i]:
            wrong_images_indexes.append(i)
    return wrong_images_indexes






#=================================================
# Plot AUC
#=================================================


def autodl_auc(outputs, targets, predictions):
    
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    
    numclass = targets.max()+1
    
    boolean_array = np.zeros((len(outputs),numclass), dtype=bool)
    
    for labelindex in range(numclass):
        boolean_array[:,labelindex]= (targets == labelindex)
    
    
    auc = roc_auc_score(boolean_array, outputs)
    auc_1 = 2*auc-1
    
    
    return round(auc, 2), round(auc_1, 2)
   


def plot_auc(outputs, targets, predictions, autodl_auc_0,autodl_auc_1, super_category, categories):
    
    print("--------------------------------------------")
    print("Average AUC")
    print("--------------------------------------------")
    
    print("AUC : ", autodl_auc_0)
    print("2*AUC-1 : ", autodl_auc_1)
    print()
    

    
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    
    
    
    numclass = len(categories) #targets.max()+1 
    
    scores_auroc = []
    scores_auroc_1 =[]

    for labelindex in range(numclass):
        binary_targets = (targets == labelindex)
        binary_predictions = (predictions == labelindex)
       
        selected_outputs = outputs[:,labelindex]
  
        auroc = roc_auc_score(binary_targets, selected_outputs)
        
        scores_auroc.append(auroc)
        scores_auroc_1.append(2*auroc-1)
        
    
    
    
    
    
    #save categories auc  
    categories_auc_textfile_path = os.path.join(PREDICTIONS_PATH, super_category, 'categories_auc.txt') 
    with open(categories_auc_textfile_path, 'w') as f:
        for i,cat in enumerate(categories):
            single_auroc = round(scores_auroc[i], 2)
            
            f.write("%s : %s\n" %(cat,single_auroc))
    
    
    
    
    
    
    average_auc = round(np.mean(scores_auroc), 2)
    average_auc_1 = round(np.mean(scores_auroc_1), 2)
    if average_auc_1 == 0.0:
        average_auc_1 = 0
    
    
    
    # Plot AUC Score
    ymin = np.min(scores_auroc_1) if np.min(scores_auroc_1) < 0  else 0
    width = 0.2
    
    fig = plt.figure(figsize=(3*numclass,8))
    
    
    
    plt.bar(np.arange(numclass), scores_auroc, width,  label='AUC')
    plt.bar(np.arange(numclass)+width, scores_auroc_1, width, label='2*AUC-1')
    plt.hlines(y=0.0, xmin=-width, xmax=numclass-1+width*2, linewidth=1, linestyles='-', color='black')
    plt.hlines(y=average_auc, xmin=-width, xmax=numclass-1+width*2, linewidth=2, linestyles='--', color='b', label='Average AUC : %0.2f'%average_auc)
    plt.hlines(y=average_auc_1, xmin=-width, xmax=numclass-1+width*2, linewidth=2, linestyles='--', color='r', label='Average 2*AUC-1 : %0.2f'%average_auc_1)
    plt.title('AUC Score - ' + super_category, fontsize=20)
    plt.ylabel('AUC Score', fontsize=16)
    plt.xlabel('Classes', fontsize=16)
    plt.ylim(ymin,1)
    plt.xticks(np.arange(numclass) + width / 2, np.arange(numclass))
    plt.legend()
#     plt.show()
    plt.pause(0.1)
    
    super_category_path = os.path.join(PREDICTIONS_PATH,super_category, "auc.png")
    fig.savefig(super_category_path, dpi=fig.dpi)
    
    
    
    
    # histogram
    fig = plt.figure(figsize=(8,5))
    plt.title('AUC Histogram', fontsize=20)
    plt.xlabel('AUC Score', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.hist(scores_auroc, alpha=0.5, ec='black')
    plt.pause(0.1)
    super_category_path = os.path.join(PREDICTIONS_PATH,super_category, "auc_histogram.png")
    fig.savefig(super_category_path, dpi=fig.dpi)
    
    

    
    
    
    

#=================================================    
# Plot ROC Curves
#=================================================

def plot_roc_curves(train_outputs, train_targets, train_predictions, valid_outputs, valid_targets, valid_predictions, super_category):
    
    print("--------------------------------------------")
    print("ROC Curves")
    print("--------------------------------------------")
    
    
   
    train_targets = np.asarray(train_targets)
    train_predictions = np.asarray(train_predictions)
    valid_targets = np.asarray(valid_targets)
    valid_predictions = np.asarray(valid_predictions)
    
    
    numclass = train_predictions.max()+1 
    
    
    train_auc_curves = []
    valid_auc_curves = []

    for labelindex in range(numclass):
        train_binary_targets = (train_targets == labelindex)
        train_binary_predictions = (train_predictions == labelindex)
        
        valid_binary_targets = (valid_targets == labelindex)
        valid_binary_predictions = (valid_predictions == labelindex)
       
        selected_train_outputs = train_outputs[:,labelindex]
        selected_valid_outputs = valid_outputs[:,labelindex]
        
        train_fpr, train_tpr, _ = roc_curve(train_binary_targets, selected_train_outputs)
        valid_fpr, valid_tpr, _ = roc_curve(valid_binary_targets, selected_valid_outputs)
       
        train_auc_curves.append([train_fpr,train_tpr])
        valid_auc_curves.append([valid_fpr,valid_tpr])
        
  
    
    
    
     # Plot ROC Curves

   
    fig = plt.figure(figsize=(20,8))
   
    plt.subplot(1,2,1)
    for idx, auc_cur in enumerate(train_auc_curves): 
        plt.plot(auc_cur[0], auc_cur[1], marker='.',  label='Class:'+str(idx))
    plt.plot([0,1], [0,1], linestyle='--', color='black')
    plt.title('Train ROC Curves - ' + super_category, fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.legend()
    
    
    plt.subplot(1,2,2)
    for idx, auc_cur in enumerate(valid_auc_curves): 
        plt.plot(auc_cur[0], auc_cur[1], marker='.',  label='Class:'+str(idx))
    plt.plot([0,1], [0,1], linestyle='--', color='black')
    plt.title('Valid ROC Curves - ' + super_category, fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.legend()
    
    plt.pause(0.1)
    
    super_category_path = os.path.join(PREDICTIONS_PATH,super_category, "roc_curves.png")
    fig.savefig(super_category_path, dpi=fig.dpi)
    


    
    

#=================================================    
# Save Predictions
#=================================================

def save_predictions(super_category, categories, training_result):

    train_dic_for_df = {}
    valid_dic_for_df = {}


    train_dic_for_df['ground_truth'] = training_result['train_ground']
    train_dic_for_df['predictions'] = training_result['train_predictions']
    
    valid_dic_for_df['ground_truth'] = training_result['valid_ground']
    valid_dic_for_df['predictions'] = training_result['valid_predictions']

    train_probability_array = list(training_result['train_predicted_probabilities'].T)
    valid_probability_array = list(training_result['valid_predicted_probabilities'].T)
    for idx, prob in enumerate(train_probability_array):
        key = 'prob_'+str(idx)
        train_dic_for_df[key] = prob
    for idx, prob in enumerate(valid_probability_array):
        key = 'prob_'+str(idx)
        valid_dic_for_df[key] = prob


    train_df = pd.DataFrame(train_dic_for_df)
    valid_df = pd.DataFrame(valid_dic_for_df)
    

    
    csv_train = os.path.join(PREDICTIONS_PATH, super_category, 'train.csv') 
    csv_valid = os.path.join(PREDICTIONS_PATH, super_category, 'valid.csv') 
    

    #save train valid_predictions_ground_probabilities in CSV  
    train_df.to_csv(csv_train, index=False)
    valid_df.to_csv(csv_valid, index=False)

    
    #save categories  
    categories_textfile_path = os.path.join(PREDICTIONS_PATH, super_category, 'categories.txt') 
    with open(categories_textfile_path, 'w') as f:
        for item in categories:
            f.write("%s\n" % item)
    
    #save category logs
    category_logfile_path = os.path.join(PREDICTIONS_PATH, super_category, 'logs.txt') 
    with open(category_logfile_path, 'w') as f:
        f.write("Total Images : %s\n" % training_result['total_images'])
        f.write("Training Images : %s\n" % training_result['train_images'])
        f.write("Validation Images : %s\n" % training_result['valid_images'])
        f.write("Training Time : %s\n" % training_result['training_time'])
        f.write("Best Train Accuracy : %s\n" % training_result['train_best_score'])
        f.write("Best Valid Accuracy : %s\n" % training_result['valid_best_score'])
        f.write("AUC : %s\n" % training_result['AUC'])
        f.write("2*AUC-1 : %s\n" % training_result['AUC_1'])
        f.write("Standard Error : %s\n" % training_result['standard_error'])
        

    
    print("Saved Results for Super-Category : ", super_category)
    print()
    print()
        
    #save super-category
    super_categories_textfile_path = os.path.join(PREDICTIONS_PATH, 'super_categories.txt') 
    with open(super_categories_textfile_path, 'a') as f:
        f.write("%s\n" % super_category)
        


        
        

#=================================================
# Save Logs
#=================================================
def save_overall_logs():
    #save logs 
    overall_logsfile_path = os.path.join(PREDICTIONS_PATH, 'logs.txt') 
    with open(overall_logsfile_path, 'w') as f:
        f.write("Total Super Categories : %s\n" % total_super_categories)
        f.write("Total Categories : %s\n" % total_categories)
        f.write("Iterations Needed : %s\n" % iterations_needed)
        f.write("Classes to Combine : %s\n" % CATEGORIES_TO_COMBINE)
        
    print("#################################")
    print("Saved Overall logs of experiment")
    print("#################################")
        
        
        
def make_super_category_directory(super_category):
    
    super_category_path = os.path.join(PREDICTIONS_PATH,super_category)
    if not os.path.exists(super_category_path):
        os.makedirs(super_category_path)
        
    



    
    
    
    
#=================================================    
# Generate Overall Histogram
#=================================================

def get_range(dictionary, begin, end):
    return dict(e for i, e in enumerate(dictionary.items()) if begin <= i <= end)


def generate_overall_auc_histogram_and_desc_auc_plot():
    
    #Read super_Categories
    super_categories_textfile_path = os.path.join(PREDICTIONS_PATH, 'super_categories.txt') 
    with open(super_categories_textfile_path, 'r') as f:
        super_categories = f.read().splitlines()
        
    #Read Categories AUC    
    categories_dic = {}
    for sup_cat in super_categories:
        categories_auc_textfile_path = os.path.join(PREDICTIONS_PATH, sup_cat, 'categories_auc.txt') 
        with open(categories_auc_textfile_path, 'r') as f:
            categories_auc = f.read().splitlines()
        for item in categories_auc:
            arr_split = item.split(' : ')
            categories_dic[arr_split[0]] = float(arr_split[1])
            
    #Sort in descending order        
    sorted_categories_dic = dict(sorted(categories_dic.items(), key=lambda x: x[1], reverse=True))
    
    
    print("##########################################")
    print("Saving Overall AUC Histogram")
    print("##########################################")
    print()
    print()
    
    #plot histogram
    fig = plt.figure(figsize=(15,8))
    plt.title('All CategoriesAUC Histogram', fontsize=20)
    plt.xlabel('AUC Score', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.hist(list(sorted_categories_dic.values()), alpha=0.5, ec='black')
    plt.show()
    overall_categoris_auc_histogram_path = os.path.join(PREDICTIONS_PATH, "overall_auc_histogram.png")
    fig.savefig(overall_categoris_auc_histogram_path, dpi=fig.dpi)
    
    
    
    
    
    print("##########################################")
    print("Saving Desc AUC Plot")
    print("##########################################")
    print()
    print()
    
    total_cat = len(sorted_categories_dic)
    cat_per_plot = 30
    plots_needed = math.ceil(len(sorted_categories_dic)/cat_per_plot)

    print("Total Categories : ", total_cat)
    print("Plots Needed : ", plots_needed)
    print("Categories per plot : ", cat_per_plot)
    
    fig, axs = plt.subplots(plots_needed,1, figsize=(20, plots_needed*cat_per_plot))
    fig.subplots_adjust(hspace = .1, wspace=.001)
    axs = axs.ravel()

    for i in range(plots_needed):
        begin = i*cat_per_plot
        end = (i*cat_per_plot+cat_per_plot-1) if (i*cat_per_plot+cat_per_plot-1) < total_cat else total_cat 
        slice_i = get_range(sorted_categories_dic, begin, end)

        axs[i].barh(list(slice_i.keys()), list(slice_i.values()))
        axs[i].set_title("All Categories AUC - Slice "+ str(i+1), fontsize=20)
        axs[i].xaxis.set_tick_params(rotation=90)
        axs[i].set_xlabel("AUC Score")
        axs[i].set_ylabel("Category")
        axs[i].invert_yaxis()
        ymin,ymax=axs[i].get_ylim()
        axs[i].vlines(0.5, ymin=ymin, ymax=ymax, linestyles ="--", colors ="r")


    descending_categoris_auc_path = os.path.join(PREDICTIONS_PATH, "descending_auc.png")
    fig.savefig(descending_categoris_auc_path, dpi=50)
    
    
    
    
    
    
    

#=================================================
#=================================================
# Magic Happens here
#=================================================
#=================================================

def load_images(single_super_data):
        
    train_images = []
    valid_images = []
    
    for image_name in single_super_data['train_data']:
        file = IMAGE_PATH+"/"+image_name
        img = cv2.imread(file)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        train_images.append(img_rgb)

    for image_name in single_super_data['valid_data']:
        file = IMAGE_PATH+"/"+image_name
        img = cv2.imread(file)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        valid_images.append(img_rgb)
        
    single_super_data['train_images'] = train_images
    single_super_data['valid_images'] = valid_images
    return single_super_data



def train_single_super_category(single_super_data):
    
    single_super_data = load_images(single_super_data)
    

    dataloaders, dataset_sizes, data_stats = make_dataset(single_super_data)
    model = getModel(number_of_classes=len(single_super_data['categories']))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    result = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, exp_lr_scheduler, num_epochs=10)
    
    
    #data statistics
    result['total_images'] = data_stats['total_images']
    result['train_images'] = data_stats['train_images']
    result['valid_images'] = data_stats['valid_images']
    
    
    
    # make directory for storing results
    make_super_category_directory(single_super_data['super_category'])

    #Accuracy, Loss and Error
    result['standard_error'] = get_error_bar(result['valid_best_score'], len(single_super_data['valid_images']))
    plot_train_results(result, single_super_data['super_category'])
    
        
    #Confusion Matrix
    plot_confusion_matrix(result['valid_ground'], result['valid_predictions'], single_super_data['super_category'], single_super_data['categories'])
    
    #AUC
    autodl_auc_0 , autodl_auc_1 = autodl_auc(result['valid_predicted_probabilities'], result['valid_ground'], result['valid_predictions'])
    result['AUC'] = autodl_auc_0
    result['AUC_1'] = autodl_auc_1
    plot_auc(result['valid_predicted_probabilities'], result['valid_ground'], result['valid_predictions'],autodl_auc_0, autodl_auc_1, single_super_data['super_category'], single_super_data['categories'])
    
    
    #ROC Curves
    plot_roc_curves(result['train_predicted_probabilities'], result['train_ground'], result['train_predictions'],
            result['valid_predicted_probabilities'], result['valid_ground'], result['valid_predictions']
            , single_super_data['super_category'])
    
    #Sample Images
    plot_sample_images(single_super_data['super_category'], result['train_ground'], single_super_data['train_images'])
    
    #Wrong Images
    plot_wrongly_classified_images(single_super_data['super_category'], result['valid_ground'], result['valid_predictions'], single_super_data['valid_images'])
    
    #Save Predictions
    save_predictions(single_super_data['super_category'], single_super_data['categories'], result)

    
    
    #CleanUp
    del dataloaders
    del model
    del criterion
    del optimizer
    del exp_lr_scheduler
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return result





#=================================================
# Training starts here
#=================================================
super_results = {}


# save overall logs
save_overall_logs()

# loop over all random super-categories to get results per super-category
for index, singlee_super_data in enumerate(super_data):
    if MAX_EPISODES is None or index < MAX_EPISODES:
        super_results[singlee_super_data['super_category']] = train_single_super_category(singlee_super_data)
        plt.close('all')

        
        
#generate over all hostogram and auc desc plot
generate_overall_auc_histogram_and_desc_auc_plot()



#Finish
print("The whole process done in {:.2f} s.".format(time.time() - t0_start))





