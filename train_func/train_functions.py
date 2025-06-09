import torch
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import os

def train_model(model,\
                train_loader, val_loader, \
                criterion, optimizer, \
                num_epochs=25, device='cuda', \
                patience=5, tolerence=0.05,\
                scheduler=None, cutmix_or_mixup=None,NUM_CLASSES=4,\
                param_str = 'best_model'):
    train_loss_history = []
    val_loss_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    early_stop_counter = 0  # Counter for early stopping

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
                cutmix_or_mixup_loop = cutmix_or_mixup 
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader
                cutmix_or_mixup_loop = None

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloader:

                inputs, labels = inputs.to(device), labels.to(device)

                if cutmix_or_mixup_loop is not None and phase == 'train':
                    inputs, labels = cutmix_or_mixup_loop(inputs, labels)
                else:
                    labels = torch.nn.functional.one_hot(labels, num_classes=NUM_CLASSES).float()
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    # Use criterion that supports soft labels for MixUp/CutMix
                    loss = criterion(outputs, labels)


                    # Backward pass and optimize in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == torch.argmax(labels, dim=1).data)


            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Record the history
            if phase == 'train':
                train_loss_history.append(epoch_loss)
            else:
                val_loss_history.append(epoch_loss)

                # Deep copy the model if the validation accuracy is improved
                if epoch_acc > best_acc - tolerence:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    early_stop_counter = 0  # Reset the early stop counter if validation improves
                else:
                    early_stop_counter += 1  # Increment the early stop counter if no improvement

            # Learning Rate Scheduler Step
            if scheduler and phase == 'val':
                # Adjust the learning rate based on the validation loss
                scheduler.step(epoch_loss)

        # Check for early stopping condition ***
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation accuracy for {patience} consecutive epochs.")
            break

    print(f'Best val Acc: {best_acc:4f}')

    # Load and save best model weights
    # model.load_state_dict(best_model_wts)
    model = save_best_model(param_str, model, best_model_wts)
    
    return model, train_loss_history, val_loss_history, best_acc

def save_best_model(param_str, model, best_model_wts):
    # Define the directory path for saving the model weights
    save_dir = './model_weights'

    # Check if the directory exists, if not, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create the full path to save the model weights
    save_path = os.path.join(save_dir, f'{param_str}.pth')

    # Load the best model weights into the model
    model.load_state_dict(best_model_wts)

    # Save the model weights
    torch.save(model.state_dict(), save_path)

    print(f"\nModel weights saved at {save_path}\n")
    return model

# Step 2: Define the evaluation function
# added a param: model_name
def evaluate_model(model, dataloader, device='cuda', model_name="missing"):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    # all_labels = y_true
    # all_preds = y_prediced
    # all_probs = y_predicted_prbability

    # calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    print("Evaluation Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"\nConfusion Matrix:\n {cm}")              # print cm here
    print("\nClassification Report:\n", classification_report(all_labels, all_preds))


    # calculate ROC curve and plot
    all_labels = np.array(all_labels)                                           # convert arrays to numpy.ndarray
    all_probs = np.array(all_probs)
    
    n_classes = all_probs.shape[1]                                               # identify the number of classes
    y_true_bin = label_binarize(all_labels, classes = list(range(n_classes)))    # turn true label into one-hot label

    plt.figure()
    # plot roc curve for each class
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probs[:,i])
        auc = roc_auc_score(y_true_bin[:, i], all_probs[:,i])
        plt.plot(fpr, tpr, label =f'Class {i} (AUC = {auc:.4f})')

    plt.plot([0, 1], [0, 1], linestyle = '--', color = "grey")         # diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Multiclass ROC for {model_name}")
    plt.legend(loc = "lower right")
    plt.grid(True)
    # save plot to png file
    plt.savefig(f"ROC_curve_{model_name}.png")
    plt.show()

    # return
    return acc, auc, precision, recall, f1, cm, fpr, tpr            # return ?

def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False