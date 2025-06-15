#%%
from dataset import build_dataloader
from models import models
from train_func import train_model, evaluate_model, set_seed
from utils import get_valid_classes
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision.transforms import v2
import optuna
import os
#%%
# root_dir = '/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/eardrumDs_kaggle'
# split_ratio=(0.70, 0.15)
# batch_size=32
# num_workers=4
# lr = 0.001
# weight_decay=0.0

# num_epochs = 25
# patience=5
# tolerence=0.05
# # momentum=0.9

# scheduler_flag = True
# factor_schedule=0.1
# patience_schedule=3

# weight_loss_flag = True
# mixup_flag = True

# testing if path exists
print("\nRunning on:", os.uname().nodename)
print("Can see data?", os.path.exists("/isilon/datalake/gurcan_rsch/scratch/otoscope/Hao/compare_frame_selection/data/Auto_selected_new_all"), "\n")

# problem reasoning: the data folder can only be accessed through certain partition & node, such as ciaq

# -----------------------------------------testing root_dir-----------------------------------------------------
 # auto selected frames - 4 classes, take care
root_dir = '/isilon/datalake/gurcan_rsch/scratch/otoscope/Hao/compare_frame_selection/data/Auto_selected_new_all'


# human selected frames - 4 classes
# root_dir = '/isilon/datalake/gurcan_rsch/scratch/otoscope/Hao/compare_frame_selection/data/human_selected_new_all'

# ---------------------------------------------------------------------------------------------------------------
def objective(trial):
    split_ratio=(0.70, 0.15)
    scheduler_flag = False
    num_workers = 1
    patience = 5
    tolerence = 0.05
    factor_schedule=0.1
    patience_schedule=3
    batch_size = trial.suggest_int('batch_size', 8, 128)
    lr = trial.suggest_loguniform('lr', 1e-6,1e-2)
    num_epochs = trial.suggest_int('num_epochs', 25, 100)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-4,5e-1)
    weight_loss_flag = trial.suggest_categorical('weight_loss_flag', [True, False])
    mixup_flag = trial.suggest_categorical('mixup_flag', [True, False])

    # root_dir = '/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/2019_2021/All_Selected_Still_Frames/All_Selected_Still_Frames'

    # print("Trial running with root_dir:", root_dir)      # debug
    # print("Exists?", os.path.exists(root_dir))
    set_seed(42)

    train_loader, val_loader, test_loader,valid_classes, class_counts  = build_dataloader(root_dir, split_ratio=split_ratio, batch_size=batch_size, num_workers=num_workers)

    if mixup_flag:
        NUM_CLASSES = len(valid_classes)
        cutmix = v2.CutMix(num_classes=NUM_CLASSES)
        mixup = v2.MixUp(num_classes=NUM_CLASSES)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    else:
        cutmix_or_mixup = None

    if weight_loss_flag:
        total_samples = sum(class_counts.values())
        # Step 2: Calculate class weights: total_samples / (number of samples in each class)
        class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
        weights = [class_weights[cls] for cls in sorted(class_counts.keys())]
        weights_tensor = torch.tensor(weights, dtype=torch.float).to('cuda')
    else:
        weights_tensor = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model: ResNet34 (adjusted for number of classes)
    # num_classes = len(valid_classes)  # Assuming valid_classes are defined as in the previous example
    # model_name = 'ResNet34'
    # model = models.resnet34(weights = "IMAGENET1K_V1")
    # model.fc = nn.Linear(model.fc.in_features, num_classes)
    # model = model.to(device)

    # Model: EfficientNet
    # num_classes = len(valid_classes)  # Assuming valid_classes are defined as in the previous example
    # model_name = "efficientnetb0"
    # model = models.efficientnet_b0(weights = "IMAGENET1K_V1")
    # model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    # model = model.to(device)

    # Model: ConvNeXt
    num_classes = len(valid_classes)
    model_name = "convnext"
    model = models.get_convnext(size = model_size, weights = "DEFAULT", num_classes = num_classes)
    model = models.convnext(weights = models.ConvNeXt_Tiny_Weights.DEFAULT)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    model = model.to(device)

    # Define the criterion, optimizer, and scheduler
    if mixup_flag:
        criterion = nn.BCEWithLogitsLoss(weight=weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Define the scheduler (e.g., ReduceLROnPlateau)
    if scheduler_flag:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor_schedule, patience=patience_schedule, verbose=True)
    else:
        scheduler = None

    param_str = f'{model_name}_bs_{batch_size}_lr_{lr}_epoch_{num_epochs}_wd_{weight_decay}_wlf_{weight_loss_flag}'
    # Train the model
    trained_model, train_loss, val_loss, best_acc = train_model(model, \
                                                    train_loader, val_loader,\
                                                    criterion, optimizer,\
                                                    num_epochs, device=device, \
                                                    patience=patience, tolerence=tolerence, \
                                                    scheduler=scheduler, cutmix_or_mixup=cutmix_or_mixup,\
                                                    NUM_CLASSES = num_classes, param_str=param_str)
    save_path = f'./model_weights/{param_str}.pth'
    return best_acc, save_path

# Track all saved model paths
saved_model_paths = []

def objective_wrapper(trial):
    best_acc, save_path = objective(trial)
    saved_model_paths.append(save_path)
    return best_acc

study = optuna.create_study(direction='maximize')

### testing ###
study.optimize(objective_wrapper, n_trials=50, timeout = 3600)                    # to prevent infinite running

# Get the best parameters
best_params = study.best_params

# Rebuild the param_str for the best model using the best_params
best_param_str = f"efficientNet_bs_{best_params['batch_size']}_lr_{best_params['lr']}_epoch_{best_params['num_epochs']}_wd_{best_params['weight_decay']}_wlf_{best_params['weight_loss_flag']}"

# Best model path
best_model_path = os.path.join('./model_weights', f'{best_param_str}.pth')

# # Check if the best model file exists
# if os.path.exists(best_model_path):
#     # Delete all other models except the best one
#     for path in saved_model_paths:
#         if path != best_model_path:
#             if os.path.exists(path):
#                 os.remove(path)
#                 print(f"Deleted {path}")
# else:
#     print(f'Can\'t find the best model {best_model_path}')

print(f"Best parameters: {study.best_params}")
print(f"Best score: {study.best_value}")
print(f"Best model is saved at: {best_model_path}")
# %%
