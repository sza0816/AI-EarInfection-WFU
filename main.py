#%%
import argparse
from dataset import build_dataloader
# from models import models
from models.my_models import get_model
from train_func import train_model, evaluate_model, set_seed
from utils import get_valid_classes
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision.transforms import v2
#%%

# parse command line arguments, get model name
parser = argparse.ArgumentParser()
parser.add_argument('--model', type = str, required = True, \
                                choices=['resnet34', 'efficientnetb0', 'convnext', 'swint', 'vitbase16', 'efficientvitb0'])
parser.add_argument('--keyframe', type=str, required=True, \
                        choices=['auto', 'human']) 
args = parser.parse_args()
model_name = args.model
keyframe_mode = args.keyframe

print(f"Training {model_name} on {keyframe_mode}_selected frames\n")


# root_dir = '/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/2019_2021/All_Selected_Still_Frames/All_Selected_Still_Frames'

if keyframe_mode == 'auto':
    root_dir = '/isilon/datalake/gurcan_rsch/scratch/otoscope/Hao/compare_frame_selection/data/Auto_selected_new_all'
elif keyframe_mode == 'human':
    root_dir = '/isilon/datalake/gurcan_rsch/scratch/otoscope/Hao/compare_frame_selection/data/human_selected_new_all'

# in need: distinguish params between human & auto
# key hyperparameters to tune               # currently using human_selected, 4 classes

hyperparams = {
    'resnet34':{
        'human':{'batch_size': 49, 'lr': 4.18e-04, 'num_epochs': 45, 'weight_decay': 0.00285, 'weight_loss_flag': False, 'mixup_flag': False}, # 
        'auto':{'batch_size': 47, 'lr': 4.5e-04, 'num_epochs': 98, 'weight_decay': 0.068, 'weight_loss_flag': False, 'mixup_flag': False}, # 
        'seed': 42
    },
    'efficientnetb0':{
        # 'human':{'batch_size': 97, 'lr': 6e-04, 'num_epochs': 73, 'weight_decay': 0.00134, 'weight_loss_flag': True, 'mixup_flag': True},  # 
        # 'human':{'batch_size': 79, 'lr': 0.00042, 'num_epochs': 78, 'weight_decay': 0.0016, 'weight_loss_flag': True, 'mixup_flag': True},  # 200 tune, 0.6957
        # 'human':{'batch_size': 82, 'lr': 0.0011, 'num_epochs': 97, 'weight_decay': 0.0262, 'weight_loss_flag': True, 'mixup_flag': False},  # 200 tune, 0.6087
        # 'human':{'batch_size': 79, 'lr': 0.0006, 'num_epochs': 78, 'weight_decay': 0.0014, 'weight_loss_flag': True, 'mixup_flag': True},  # 0.6739
        'human':{'batch_size': 79, 'lr': 0.0007, 'num_epochs': 78, 'weight_decay': 0.0014, 'weight_loss_flag': True, 'mixup_flag': True},  # 0.7826

        # 'auto':{'batch_size': 114, 'lr': 0.0012, 'num_epochs': 93, 'weight_decay': 0.00088, 'weight_loss_flag': True, 'mixup_flag': True}, # 200 tune, 0.6087
        # 'auto':{'batch_size': 102, 'lr': 0.00103, 'num_epochs': 95, 'weight_decay': 0.0013, 'weight_loss_flag': True, 'mixup_flag': True}, # 200 tune, 0.74
        'auto':{'batch_size': 102, 'lr': 0.00103, 'num_epochs': 95, 'weight_decay': 0.0011, 'weight_loss_flag': True, 'mixup_flag': True}, # 0.74
        # 'auto':{'batch_size': 102, 'lr': 0.00095, 'num_epochs': 95, 'weight_decay': 0.0012, 'weight_loss_flag': True, 'mixup_flag': True}, # 0.70
        # 'auto':{'batch_size': 96, 'lr': 0.00103, 'num_epochs': 95, 'weight_decay': 0.0013, 'weight_loss_flag': True, 'mixup_flag': True}, # 0.7391, lower auc



        'seed': 123

    },
    'convnext':{
        'human':{'batch_size': 37, 'lr': 5e-05, 'num_epochs': 100, 'weight_decay': 0.0001, 'weight_loss_flag': False, 'mixup_flag': True}, # 
        'auto':{'batch_size': 9, 'lr': 3.7e-05, 'num_epochs': 82, 'weight_decay': 0.00192, 'weight_loss_flag': False, 'mixup_flag': False}, # 
        'seed': 42
    },
    'swint':{
        'human':{'batch_size': 58, 'lr': 2e-05, 'num_epochs': 74, 'weight_decay': 0.00014, 'weight_loss_flag': False, 'mixup_flag': True}, #
        'auto':{'batch_size': 21, 'lr': 2.1e-04, 'num_epochs': 79, 'weight_decay': 0.000186, 'weight_loss_flag': True, 'mixup_flag': False}, # 
        'seed': 42
    },
    'vitbase16':{
        # 'human':{'batch_size': 60, 'lr': 2e-05, 'num_epochs': 60, 'weight_decay': 0.01, 'weight_loss_flag': False, 'mixup_flag': False}, # 1. increase lr/wd, flags false, 0.6522
        'human':{'batch_size': 60, 'lr': 3e-05, 'num_epochs': 50, 'weight_decay': 0.01, 'weight_loss_flag': False, 'mixup_flag': False}, # 2. increase lr, decrease epoch, 0.6522
        # 'human':{'batch_size': 60, 'lr': 3e-05, 'num_epochs': 50, 'weight_decay': 0.01, 'weight_loss_flag': True, 'mixup_flag': False}, # 3. turn on weight loss flag, 0.6087
        # 'human':{'batch_size': 60, 'lr': 3e-05, 'num_epochs': 50, 'weight_decay': 0.01, 'weight_loss_flag': False, 'mixup_flag': True}, # 4. turn on mixup, 0.6087

        'auto':{'batch_size': 69, 'lr': 6e-05, 'num_epochs': 34, 'weight_decay': 0.0855, 'weight_loss_flag': True, 'mixup_flag': True}, # **0.6739**
        'seed': 42
    }, 
    'efficientvitb0':{
        # 'human':{'batch_size': 80, 'lr': 1.65e-05, 'num_epochs': 99, 'weight_decay': 0.0166, 'weight_loss_flag': False, 'mixup_flag': False}, # original **0.6739**
        # 'human':{'batch_size': 80, 'lr': 1.65e-05, 'num_epochs': 99, 'weight_decay': 0.012, 'weight_loss_flag': False, 'mixup_flag': False}, # decrease wd, 0.6739
        # 'human':{'batch_size': 80, 'lr': 1.65e-05, 'num_epochs': 120, 'weight_decay': 0.0166, 'weight_loss_flag': False, 'mixup_flag': False}, # increase epoch, 0.6522
        'human':{'batch_size': 80, 'lr': 1.65e-05, 'num_epochs': 70, 'weight_decay': 0.0166, 'weight_loss_flag': False, 'mixup_flag': False}, # decrease epoch, 0.6739
        # 'human':{'batch_size': 80, 'lr': 1.65e-05, 'num_epochs': 70, 'weight_decay': 0.012, 'weight_loss_flag': False, 'mixup_flag': False}, # decrease wd + epoch, 0.6522

        # 'auto':{'batch_size': 100, 'lr': 5e-05, 'num_epochs': 100, 'weight_decay': 0.000135, 'weight_loss_flag': True, 'mixup_flag': True} # original **0.6087**
        'auto':{'batch_size': 80, 'lr': 1.65e-05, 'num_epochs': 99, 'weight_decay': 0.0166, 'weight_loss_flag': False, 'mixup_flag': False}, # 1. try same params, 0.6522
        # 'auto':{'batch_size': 80, 'lr': 1.65e-05, 'num_epochs': 99, 'weight_decay': 0.01, 'weight_loss_flag': False, 'mixup_flag': False}, # 2. decrease wd, 0.6304
        # 'auto':{'batch_size': 80, 'lr': 2e-05, 'num_epochs': 99, 'weight_decay': 0.0166, 'weight_loss_flag': False, 'mixup_flag': False}, # 3. increase lr, 0.6304
        # 'auto':{'batch_size': 80, 'lr': 1.65e-05, 'num_epochs': 99, 'weight_decay': 0.0166, 'weight_loss_flag': True, 'mixup_flag': False}, # 4. wlf = T, 0.5870
        'seed': 42
    }
}

params = hyperparams[model_name][keyframe_mode]

batch_size = params['batch_size']
lr = params['lr']
num_epochs = params['num_epochs']
weight_decay = params['weight_decay']
weight_loss_flag = params['weight_loss_flag']
mixup_flag = params['mixup_flag']


# other hyperparameters
split_ratio=(0.7, 0.15)
num_workers=1
patience=5             # for early stopping
tolerence=0.05
momentum=0.9

scheduler_flag = False
factor_schedule=0.1
patience_schedule=3

seed = hyperparams[model_name]['seed']
set_seed(seed)

# split data for modeling
train_loader, val_loader, test_loader,valid_classes, class_counts  = build_dataloader(root_dir, \
                                                                                      split_ratio=split_ratio, \
                                                                                      batch_size=batch_size, \
                                                                                      num_workers=num_workers)

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


# Use GPU if available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
num_classes = len(valid_classes)
model = get_model(model_name, num_classes, 'DEFAULT')
model = model.to(device)

if mixup_flag:
    criterion = nn.BCEWithLogitsLoss(weight=weights_tensor)        # if mixup, use binary cross entropy with logits loss function
                                                                   # suitable for soft label or multi-label tasks
else:
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)         # if no mixup, use cross entropy loss function

# Adam optimizer
optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)       

# Define the scheduler (e.g., ReduceLROnPlateau)
if scheduler_flag:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, \
                                                    mode='min', \
                                                    factor=factor_schedule, \
                                                    patience=patience_schedule)               # verbose = T is no longer used
# adjust learning rate (lr) based on training result
else:
    scheduler = None

param_str = f'{model_name}_bs_{batch_size}_lr_{lr}_epoch_{num_epochs}_wd_{weight_decay}_wlf_{weight_loss_flag}'
# print(f"{param_str}\n")

# Train the model
trained_model, train_loss, val_loss, best_acc = train_model(model, \
                                                train_loader, val_loader,\
                                                criterion, optimizer,\
                                                num_epochs, device=device, \
                                                patience=patience, tolerence=tolerence, \
                                                scheduler=scheduler, cutmix_or_mixup=cutmix_or_mixup,\
                                                NUM_CLASSES = num_classes, param_str = param_str
                                                )

# Plot training & validation loss
plt.figure()
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title(f'Training & Validation Loss ({model_name})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"output_{model_name}/{keyframe_mode}/T_V_loss_{model_name}.png")  # Save the figure before plt.show(
plt.show()

# Evaluate the model on the test set
print("Evaluating on the test set...")
evaluate_model(trained_model, test_loader, device = device, model_name = model_name, mode = keyframe_mode)