#%%
from dataset import build_dataloader
from models import models
from train_func import train_model, evaluate_model, set_seed
from utils import get_valid_classes
import torch
import torch.nn as nn
from torch.optim import AdamW, Adam
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision.transforms import v2
#%%

# root_dir = '/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/2019_2021/All_Selected_Still_Frames/All_Selected_Still_Frames'

# auto selected frames - 4 classes, take care
root_dir = '/isilon/datalake/gurcan_rsch/scratch/otoscope/Hao/compare_frame_selection/data/Auto_selected_new_all'

# human selected frames - 4 classes
# root_dir = '/isilon/datalake/gurcan_rsch/scratch/otoscope/Hao/compare_frame_selection/data/human_selected_new_all'


# hyperparameters inherited from main, adjust
split_ratio=(0.7, 0.15)
batch_size=32
num_workers=1
lr = 1.3e-04
weight_decay=0.2         # increase weight_decay

num_epochs = 30
patience=5             # for early stopping
tolerence=0.05
momentum=0.9

scheduler_flag = False
factor_schedule=0.1
patience_schedule=3


weight_loss_flag = False           # whether to apply class-balanced weighting to the loss function
mixup_flag = False                 # whether to apply data augmentation

set_seed(42)

train_loader, val_loader, test_loader,valid_classes, class_counts  = build_dataloader(root_dir, \
                                                                                      split_ratio=split_ratio, \
                                                                                      batch_size=batch_size, \
                                                                                      num_workers=num_workers)
#%%


# for reproductivity & scalability usage, see the last 2 params
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

# Model: ConvNeXt
num_classes = len(valid_classes)
model_name = "convnext"
model = models.convnext_tiny(weights = models.ConvNeXt_Tiny_Weights.DEFAULT)
model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
model.classifier = nn.Sequential(
    model.classifier[0],           # layerNorm
    nn.Flatten(1),
    nn.Dropout(p=0.3),             # add new dropout
    model.classifier[2]            # Linear
)
model = model.to(device)


if mixup_flag:
    criterion = nn.BCEWithLogitsLoss(weight=weights_tensor)        # if mixup, use binary cross entropy with logits loss function
                                                                   # suitable for soft label or multi-label tasks
else:
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)         # if no mixup, use cross entropy loss function

# AdamW optimizer
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)       

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
plt.title(f'ConvNeXt_{model_size} Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"output_{model_name}/T_V_loss_{model_name}.png")  # Save the figure before plt.show(
plt.show()

# Evaluate model on the test set
print("Evaluating on the test set...")
evaluate_model(trained_model, test_loader, device=device, model_name = model_name)