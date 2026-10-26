import sys
import os
import re

# If needed, add the parent directory of 'pointcept' to sys.path
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time
from torch.cuda.amp import autocast
import torch._dynamo
torch._dynamo.config.suppress_errors = True
# Remove the stub Point class and import the actual one

import pandas as pd
import numpy as np
import random

from custom_models import CustomPointCloudClassifier
from pointcept.models.utils.structure import Point

"""
#### NOISE FOR GENERALIZATION #######
def jitter(coords, sigma=0.01, clip=0.05):
    noise = np.clip(sigma * np.random.randn(*coords.shape), -clip, clip)
    return coords + noise

def random_translate(coords, translate_range=(-0.2, 0.2)):
    translation = np.random.uniform(translate_range[0], translate_range[1], size=(1, 3))
    return coords + translation

def random_scale(coords, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return coords * scale

def augment_point_cloud(sample):
    # Extract the coordinate tensor from the sample's data_dict
    coords_tensor = sample['data_dict']['coord']
    # Convert the tensor to a NumPy array for augmentation
    coords = coords_tensor.cpu().numpy()
    
    # Apply the augmentations: scale, translate, then jitter
    coords = random_scale(coords)            
    coords = random_translate(coords)        
    coords = jitter(coords)                  

    # Convert the augmented coordinates back to a torch tensor
    augmented_coords = torch.from_numpy(coords).float()
    
    # Update the sample with the augmented coordinates
    sample['data_dict']['coord'] = augmented_coords

    # Optionally, update the 'feat' tensor if it depends on coordinates.
    # For example, if 'feat' is a concatenation of coords and intensity, you might do:
    # intensity = sample['data_dict']['feat'][:, 3:]  # assuming intensity is at column index 3
    # sample['data_dict']['feat'] = torch.cat((augmented_coords, intensity), dim=1)
    
    return sample
"""


class PointCloudDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Folder containing ID -> class -> .csv.
                            E.g. 'train', 'val', 'test'.
            transform (callable, optional): Optional transform to apply to each sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # (file_path, label, id_str)

        # 1) Collect .csv files under root_dir -> ID -> class
        id_dirs = sorted(d for d in os.listdir(root_dir)
                         if os.path.isdir(os.path.join(root_dir, d)))
        for id_dir in id_dirs:
            id_path = os.path.join(root_dir, id_dir)
            class_dirs = sorted(d for d in os.listdir(id_path)
                                if os.path.isdir(os.path.join(id_path, d)))
            for c_dir in class_dirs:
                label = int(c_dir)  # we assume folder name is the class label
                class_path = os.path.join(id_path, c_dir)

                for file_name in os.listdir(class_path):
                    if file_name.endswith(".csv"):
                        file_path = os.path.join(class_path, file_name)
                        self.samples.append((file_path, label, id_dir))

    def __len__(self):
        return len(self.samples)
        

    def __getitem__(self, idx):
        file_path, label, id_str = self.samples[idx]
        
        # Extract frame number using regex
        file_name = os.path.basename(file_path)
        match = re.search(r'\(Frame (\d+)\)', file_name)
        if match:
            frame_num = int(match.group(1))
        else:
            frame_num = -1  # or handle the error as needed
        
        # Load CSV as before
        data = pd.read_csv(file_path, engine='python', on_bad_lines='warn')
        intensity = data.iloc[:, 0]
        coords = data.iloc[:, [7, 8, 9]].to_numpy(dtype=np.float32, copy=True)
        coords = np.ascontiguousarray(coords)
        
        intensity = pd.to_numeric(intensity, errors='coerce').values
        valid_idx = ~np.isnan(intensity)
        intensity = intensity[valid_idx] / 255.0
        coords = coords[valid_idx]
        
        intensity_tensor = torch.from_numpy(intensity).float().unsqueeze(1)
        coords_tensor = torch.from_numpy(coords).float()
        coords_tensor = coords_tensor - coords_tensor.mean(dim=0, keepdim=True)
        
        feat = torch.cat((coords_tensor, intensity_tensor), dim=1)
        N = coords_tensor.shape[0]
        batch = torch.zeros(N, dtype=torch.long)
        
        data_dict = {
            'coord': coords_tensor,
            'feat': feat,
            'batch': batch,
            'grid_size': 0.05
        }
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        sample = {
            'data_dict': data_dict,
            'label': label_tensor,
            'id': id_str,
            'frame': frame_num  # new: include frame number
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


########################################
# EarlyStopping & Training Routines
########################################
class EarlyStopping:
    """
    Early stops the training if validation accuracy doesn't improve
    after a given patience.
    """
    def __init__(self, patience=5, mode='max'):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            mode (str): 'max' to monitor accuracy, 'min' for loss, etc.
        """
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_weights = None

    def __call__(self, current_score, model, model_path='best_model.pth'):
        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(model, save_path=model_path)
        else:
            improvement = (current_score > self.best_score) if self.mode == 'max' else (current_score < self.best_score)
            if improvement:
                self.best_score = current_score
                self.save_checkpoint(model, save_path=model_path)
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

    def save_checkpoint(self, model, save_path):
        self.best_model_weights = model.state_dict()
        torch.save(self.best_model_weights, save_path)


def train_model_with_early_stopping(model, train_loader, val_loader,
                                    criterion, optimizer, device,
                                    num_epochs=10, early_stopping_patience=5,
                                    phase_tag="Phase"):
    
    '''
    total_steps = num_epochs * len(train_loader)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[0.002, 0.0002],   # For classifier and backbone respectively
        total_steps=total_steps,
        pct_start=0.04,
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=100.0
    )
    '''
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',           # since you’re monitoring validation accuracy
        factor=0.5,           # multiply LR by 0.5 when plateaued
        patience=2,           
        min_lr=1e-7
    )
    #scheduler = CosineAnnealingLR(max_T = num_epochs) #CODE WAS SOMETHING LIKE THIS 
    
    early_stopper = EarlyStopping(patience=early_stopping_patience, mode='max')

    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0

        for batch in train_loader:
            # Now each 'batch' is a dict with:
            #   batch['data_dict'] = list of data_dicts
            #   batch['label'] = stacked labels
            #   batch['id'] = list of strings
            data_dict_list = batch['data_dict']
            labels = batch['label'].to(device)
            batch_size = len(data_dict_list)
            optimizer.zero_grad()

            outputs = []
            #with autocast(dtype=torch.bfloat16):
            # Process each sample
            for i in range(batch_size):
                sample_dict = data_dict_list[i]
                # Move each tensor in sample_dict to device
                for k, v in sample_dict.items():
                    if isinstance(v, torch.Tensor):
                        sample_dict[k] = v.to(device)

                point = Point(sample_dict)
                point.serialization(order=model.backbone.order, shuffle_orders=model.backbone.shuffle_orders)
                point.sparsify()

                out = model(point)
                #print("Output tensor dtype:", out['cls_logits'].dtype)
                outputs.append(out['cls_logits'])  # shape [1, num_classes]

            # Stack all outputs => [batch_size, num_classes]
            logits = torch.cat(outputs, dim=0)
            torch.cuda.synchronize()  
            loss = criterion(logits, labels)
                
            loss.backward()
            #check to see how much gradient clipping is impacting training
            # Function to compute average gradient norm for a module
            def get_avg_grad_norm(module):
                    total_norm = 0.0
                    count = 0
                    for param in module.parameters():
                        if param.grad is not None:
                            total_norm += param.grad.norm().item()
                            count += 1
                    return total_norm / count if count > 0 else 0.0

            #classifier_grad_norm = get_avg_grad_norm(model.classifier)
            #print(f"Avg Gradient Norm for classifier: {classifier_grad_norm:.4f}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)
            optimizer.step()

            # Accuracy
            _, preds = torch.max(logits, dim=1)
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += batch_size
            #TROUBLESHOOTING
            '''
            print(f"Epoch {epoch+1}, Batch {i+1}/{batch_size}: Loss = {loss.item():.4f}")
            print('batch loss: ' + str(loss.item()))
            
            _, preds = torch.max(logits, dim=1)
            unique_preds, counts_preds = torch.unique(preds, return_counts=True)
            print("Predicted class distribution:", {int(u): int(c) for u, c in zip(unique_preds, counts_preds)})

            unique_labels, counts_labels = torch.unique(labels, return_counts=True)
            print("Ground truth distribution:", {int(u): int(c) for u, c in zip(unique_labels, counts_labels)})
            
            softmax_probs = torch.softmax(logits, dim=1)
            avg_probs = softmax_probs.mean(dim=0)
            print("Average softmax probabilities per class:", avg_probs.detach().cpu().numpy())

            print("Logits stats -- min: {:.4f}, max: {:.4f}, mean: {:.4f}".format(
            logits.min().item(), logits.max().item(), logits.mean().item()))
            '''


        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                data_dict_list = batch['data_dict']
                labels = batch['label'].to(device)
                batch_size = len(data_dict_list)

                outputs = []
                for i in range(batch_size):
                    sample_dict = data_dict_list[i]
                    for k, v in sample_dict.items():
                        if isinstance(v, torch.Tensor):
                            sample_dict[k] = v.to(device)

                    point = Point(sample_dict)
                    point.serialization(order=model.backbone.order, shuffle_orders=model.backbone.shuffle_orders)
                    point.sparsify()
                    out = model(point)
                    outputs.append(out['cls_logits'])

                logits = torch.cat(outputs, dim=0)
                val_loss = criterion(logits, labels)

                _, preds = torch.max(logits, 1)
                val_running_loss += val_loss.item() * batch_size
                val_running_corrects += torch.sum(preds == labels.data).item()
                val_total_samples += batch_size

        val_epoch_loss = val_running_loss / val_total_samples
        val_epoch_acc = val_running_corrects / val_total_samples

        print(f'[{phase_tag}] Epoch {epoch+1}/{num_epochs} - '
              f'Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f} | '
              f'Val Loss: {val_epoch_loss:.4f}, Acc: {val_epoch_acc:.4f}')

        early_stopper(val_epoch_acc, model, model_path= 'best_model_phase_' + phase_tag + '.pth')  # Check and save best model if improved
        
        if early_stopper.early_stop:  # Stop training if patience exceeded
            print("Early stopping triggered.")
            break

        # Update learning rate scheduler
        scheduler.step(val_epoch_acc)

    print(f'[{phase_tag}] Training complete.')



def test_model(model, test_loader, criterion, device):
    model.eval()
    test_running_loss = 0.0
    test_running_corrects = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            data_dict_list = batch['data_dict']  # list of data_dicts
            labels = batch['label'].to(device)   # shape [batch_size]
            batch_size = len(data_dict_list)

            outputs = []
            # Loop over each sample in the batch
            for i in range(batch_size):
                sample_dict = data_dict_list[i]
                # Move each tensor to device
                for k, v in sample_dict.items():
                    if isinstance(v, torch.Tensor):
                        sample_dict[k] = v.to(device)

                point = Point(sample_dict)
                point.serialization(order=model.backbone.order,
                                   shuffle_orders=model.backbone.shuffle_orders)
                point.sparsify()

                out = model(point)                # {'cls_logits': shape [1,num_classes]}
                outputs.append(out['cls_logits']) # collect to concatenate

            # Now stack the outputs => shape [batch_size, num_classes]
            logits = torch.cat(outputs, dim=0)

            loss = criterion(logits, labels)
            _, preds = torch.max(logits, 1)

            test_running_loss += loss.item() * batch_size
            test_running_corrects += torch.sum(preds == labels).item()

            # For confusion matrix
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    test_loss = test_running_loss / len(test_loader.dataset)
    test_acc = test_running_corrects / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')

    return all_labels, all_preds


def test_model_with_ids(model, test_loader, criterion, device):
    model.eval()
    test_running_loss = 0.0
    test_running_corrects = 0
    all_labels = []
    all_preds = []
    all_ids = []
    all_frames = []

    with torch.no_grad():
        for batch in test_loader:
            data_dict_list = batch['data_dict']
            labels = batch['label'].to(device)
            ids = batch['id']        # list of person IDs
            frames = batch['frame']  # list of frame numbers
            batch_size = len(data_dict_list)

            outputs = []
            for i in range(batch_size):
                sample_dict = data_dict_list[i]
                for k, v in sample_dict.items():
                    if isinstance(v, torch.Tensor):
                        sample_dict[k] = v.to(device)

                point = Point(sample_dict)
                point.serialization(order=model.backbone.order,
                                    shuffle_orders=model.backbone.shuffle_orders)
                point.sparsify()
                out = model(point)
                outputs.append(out['cls_logits'])

            logits = torch.cat(outputs, dim=0)
            loss = criterion(logits, labels)
            _, preds = torch.max(logits, 1)

            test_running_loss += loss.item() * batch_size
            test_running_corrects += torch.sum(preds == labels).item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_ids.extend(ids)     # person id for each sample
            all_frames.extend(frames)  # frame number for each sample

    test_loss = test_running_loss / len(test_loader.dataset)
    test_acc = test_running_corrects / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')

    return all_ids, all_frames, all_labels, all_preds



def analyze_orientation_sections(all_ids, all_frames, all_labels, all_preds):
    from collections import defaultdict
    # Group results by (person, label) tuple
    groups = defaultdict(list)
    for person, frame, true_label, pred in zip(all_ids, all_frames, all_labels, all_preds):
        groups[(person, true_label)].append((frame, true_label, pred))

    results = {}
    for key, samples in groups.items():
        # Sort by frame number
        samples_sorted = sorted(samples, key=lambda x: x[0])
        n = len(samples_sorted)
        # Determine the segment size (if n < 8, each segment will have one sample or you can adjust as needed)
        segment_size = n // 8 if n >= 8 else 1
        segment_accuracies = []

        for i in range(8):
            start = i * segment_size
            # For the final segment, include all remaining samples
            end = n if i == 7 else (i + 1) * segment_size
            segment = samples_sorted[start:end]
            if not segment:
                accuracy = None
            else:
                correct = sum(1 for (_, true, pred) in segment if true == pred)
                accuracy = correct / len(segment)
            segment_accuracies.append(accuracy)
        results[key] = segment_accuracies

    # Print the accuracy for each segment per person and mobility aid
    for (person, label), accuracies in results.items():
        print(f"Person: {person}, Mobility Aid (label): {label}")
        for i, acc in enumerate(accuracies, start=1):
            if acc is not None:
                print(f"  Segment {i}: Accuracy {acc*100:.2f}%")
            else:
                print(f"  Segment {i}: No samples")
    return results



def custom_collate_fn(batch):
    data_dict_list = []
    label_list = []
    id_list = []
    frame_list = []  # collect frame numbers

    for sample in batch:
        data_dict_list.append(sample['data_dict'])
        label_list.append(sample['label'])
        id_list.append(sample['id'])
        frame_list.append(sample['frame'])

    labels_tensor = torch.stack(label_list, dim=0)

    collated_batch = {
        'data_dict': data_dict_list,
        'label': labels_tensor,
        'id': id_list,
        'frame': frame_list  # include frames in the batch
    }
    return collated_batch

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v
    return new_state_dict

class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

def format_time(seconds):
    """Format time in seconds to a string of format 'Hh Mm Ss'."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"

def compute_per_person_accuracy(all_ids, all_labels, all_preds):
    from collections import defaultdict
    import numpy as np

    # Group predictions by person ID
    person_results = defaultdict(list)
    for pid, true, pred in zip(all_ids, all_labels, all_preds):
        person_results[pid].append((true, pred))
    
    # Calculate accuracy per person
    person_accuracy = {}
    for pid, results in person_results.items():
        correct = sum(1 for (true, pred) in results if true == pred)
        accuracy = correct / len(results)
        person_accuracy[pid] = accuracy
        print(f"Accuracy for {pid}: {accuracy*100:.2f}%")
    
    # Convert the accuracy values into a NumPy array for statistical analysis
    accuracies = np.array(list(person_accuracy.values()))
    average_accuracy = np.mean(accuracies)
    std_deviation = np.std(accuracies)
    
    print(f"\nAverage Accuracy: {average_accuracy*100:.2f}%")
    print(f"Standard Deviation: {std_deviation*100:.2f}%")
    return person_accuracy, average_accuracy, std_deviation

def directional_analysis_overall(segment_results):
    """
    Computes the overall average accuracy and standard deviation for each segment
    across all participants.
    
    Args:
        segment_results (dict):
            The results dictionary returned by analyze_orientation_sections.
            The keys are tuples (person, label) and the values are lists of segment accuracies.
    
    Returns:
        segment_summary (dict):
            A dictionary where keys are segment indices (1-8) and values are tuples of
            (average_accuracy, std_deviation) for that segment. If no samples exist for a segment,
            the value will be (None, None).
    """
    import numpy as np

    segment_summary = {}
    # There are 8 segments.
    for segment_index in range(8):
        # For each segment, collect accuracies from all participants (filter out None values)
        segment_accs = [
            acc_list[segment_index]
            for acc_list in segment_results.values()
            if acc_list[segment_index] is not None
        ]
        if segment_accs:
            avg_accuracy = np.mean(segment_accs)
            std_dev = np.std(segment_accs)
        else:
            avg_accuracy = None
            std_dev = None

        # Store using a 1-based segment index for clarity
        segment_summary[segment_index + 1] = (avg_accuracy, std_dev)

        # Print the result for the current segment
        if avg_accuracy is not None:
            print(f"Segment {segment_index + 1}: Average Accuracy = {avg_accuracy * 100:.2f}%, Standard Deviation = {std_dev * 100:.2f}%")
        else:
            print(f"Segment {segment_index + 1}: No samples available.")

    return segment_summary

########################################
# MAIN
########################################

def main():
    # Seed
    train_start_time = time.time()
    set_seed(42)
    
    classifier_lr = 0.0002 #4e-4
    classifier_wd = 0.0002 #4e-4

    backbone_lr = 0.00001 #4e-4
    backbone_wd = 0.00001 #4e-4

    batch_size_value = 8

    txt_string = f"7_class_data_training_log_classifierlr{classifier_lr}_classifierwd{classifier_wd}_backbonelr{backbone_lr}_backbonewd{backbone_wd}_bs{batch_size_value}.txt"
    tee = Tee(txt_string, "w")
    sys.stdout = tee

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ================
    # Build Datasets
    # ================
    train_dataset = PointCloudDataset(root_dir='/data/seankirkby/data/train')
    val_dataset   = PointCloudDataset(root_dir='/data/seankirkby/data/val')
    test_dataset  = PointCloudDataset(root_dir='/data/seankirkby/data/test')

    #DEBUGGING
    '''
    print("\n=== Debug: Inspect the first sample from train_dataset ===")

    sample = train_dataset[0]
    print("Label:", sample["label"].item())
    print("ID string:", sample["id"])  # or remove if you're not using the ID
    coord_tensor = sample["data_dict"]["coord"]
    feat_tensor = sample["data_dict"]["feat"]

    print(f"coord_tensor shape: {coord_tensor.shape}")  # e.g. (N, 3) if columns=7,8,9
    print("First few coords:\n", coord_tensor[:5])
    print(f"feat_tensor shape: {feat_tensor.shape}")     # e.g. (N, 4) if last col is intensity
    print("First few feats:\n", feat_tensor[:5])
    '''

    def filter_and_remap(dataset):
        keep = []
        for (fp, lbl, idd) in dataset.samples:
            #print(lbl)
            # Keep if lbl is in 0,1,2,3,6
            if lbl in [0, 1, 2, 3, 6]:
                keep.append((fp, lbl, idd))
        label_map = {0: 0, 1: 1, 2: 2, 3: 3, 6: 4}  # Remap 6 to 4
        remapped = [(fp, label_map[lbl], idd) for (fp, lbl, idd) in keep]
        dataset.samples = remapped

    #print("checking training")
    #filter_and_remap(train_dataset)
    #print("checking val")
    #filter_and_remap(val_dataset)
    #print("checking test")
    #filter_and_remap(test_dataset)

    #WEIGHTED SAMPLING FOR TRAINING SET --> INSERT train_sampler into sampler for train_loader
    # Extract the labels from the (file_path, label, id_str) tuples
    train_labels = [sample[1] for sample in train_dataset.samples]
    # Count how many samples for each class
    class_counts = Counter(train_labels)
    num_classes = len(class_counts)  
    # Assuming class_counts is a dictionary with keys as class labels and values as counts

    # class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}

    # Make sure the order of weights corresponds to your class labels. For example:
    weight_list = [1.0 / class_counts[i] for i in range(num_classes)]
    weights = torch.tensor(weight_list, dtype=torch.float) 
    #weights_tensor = torch.tensor(weights_list, dtype=torch.float32).to(device)
    
    # ================
    # DataLoaders
    # ================
    #insert sampler=train_sampler
    train_loader = DataLoader(train_dataset, batch_size=batch_size_value, shuffle=True, num_workers=4, collate_fn=custom_collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size_value, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size_value, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
    
    # ================
    # Model Setup
    # ================
    model_cfg = {
        'num_classes': 7,
        'backbone': {
            'in_channels': 4,
            'order': ['z', 'z-trans', 'hilbert', 'hilbert-trans'],
            'stride': (2, 2, 2, 2),
            'enc_depths': (2, 2, 2, 6, 2),
            'enc_channels': (32, 64, 128, 256, 512),
            'enc_num_head': (2, 4, 8, 16, 32),
            'enc_patch_size': (1024, 1024, 1024, 1024, 1024),
            'mlp_ratio': 4,
            'qkv_bias': True,
            'qk_scale': None,
            'attn_drop': 0.0,
            'proj_drop': 0.0,
            'drop_path': 0.3,
            'shuffle_orders': True,
            'pre_norm': True,
            'enable_rpe': False,
            'enable_flash': True,
            'upcast_attention': False,
            'upcast_softmax': False,
            'pdnorm_bn': False,
            'pdnorm_ln': False,
            'pdnorm_decouple': True,
            'pdnorm_adaptive': False,
            'pdnorm_affine': True,
            'pdnorm_conditions': ('nuScenes', 'SemanticKITTI', 'Waymo'),
        },
    }

    model = CustomPointCloudClassifier(
        backbone_cfg=model_cfg['backbone'],
        num_classes=model_cfg['num_classes']
    ).to(device)

    pretrained_model_path = 'model_best.pth'
    if os.path.exists(pretrained_model_path):
        state_dict = torch.load(pretrained_model_path, map_location=device)
        state_dict = remove_module_prefix(state_dict)
        model.load_state_dict(state_dict, strict=False)
        print("Loaded pre-trained weights from model_best.pth")
    else:
        print("Pre-trained model not found. Proceeding with randomly initialized weights.")

    # RECENTLY ADDED Compile the model
    # model = ForwardOnlyCompiledModel(model, compile_mode="default")
    # Loss function
    criterion = nn.CrossEntropyLoss(weight=weights.to(device)) 
    '''
    # ======================
    # Phase 1: Train only the classification head
    # ======================
    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Only classification head is trainable
    #classifier_params = list(model.classifier.parameters())
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=classifier_lr, weight_decay=classifier_wd)

    print("\n=== Phase 1: Training only the classification head ===")
    train_model_with_early_stopping(
        model, train_loader, val_loader, criterion, optimizer, device,
        num_epochs=40, early_stopping_patience=10, phase_tag="Phase1"
    )
    '''
    # Load best from Phase 1
    best_model_phase1_path = "best_model_phase_Phase1.pth"
    if os.path.exists(best_model_phase1_path):
        model.load_state_dict(torch.load(best_model_phase1_path))
        print("Loaded best Phase 1 model weights")
    else:
        print("Phase 1 best model not found.")
         
    # ======================
    # Phase 2: Fine-tune the entire model
    # ======================
    for param in model.parameters():
        param.requires_grad = True

    #optimizer = torch.optim.Adam(model.parameters(), lr=backbone_lr, weight_decay=backbone_wd)
    optimizer = torch.optim.Adam([
        {'params': model.backbone.parameters(), 'lr': backbone_lr, 'weight_decay': backbone_wd},
        {'params': model.classifier.parameters(), 'lr': classifier_lr, 'weight_decay': classifier_wd}
    ])

    print("\n=== Phase 2: Fine-tuning the entire model ===")
    train_model_with_early_stopping(
        model, train_loader, val_loader, criterion, optimizer, device,
        num_epochs=30, early_stopping_patience=6, phase_tag="Phase2"
    )
    
    # Load best from Phase 2
    best_model_phase2_path = "best_model_phase_Phase2.pth"
    if os.path.exists(best_model_phase2_path):
        model.load_state_dict(torch.load(best_model_phase2_path))
        print("Loaded best Phase 2 model weights")
    else:
        print("Phase 2 best model not found.")

    train_end_time = time.time()
    total_train_time = train_end_time - train_start_time
    print(f"TOTAL TIME TRAINING: {format_time(total_train_time)} seconds")
    # ================
    # Test
    # ================
    print("\n=== Testing the final model ===")
    #test_labels, test_preds = test_model(model, test_loader, criterion, device)

    # Ensure your test_loader is using shuffle=False so the frame order is preserved
    all_ids, all_frames, all_labels, all_preds = test_model_with_ids(model, test_loader, criterion, device)
    person_accuracy, avg_acc, std_acc = compute_per_person_accuracy(all_ids, all_labels, all_preds)

    # Analyze accuracy per eighth (each representing an orientation segment)
    segment_results = analyze_orientation_sections(all_ids, all_frames, all_labels, all_preds)
    segment_summary = directional_analysis_overall(segment_results)
    # ======================
    # Create and Save Confusion Matrix
    # ======================
    
    # Define class names based on your label mapping
    class_names = ['Class0', 'Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6']  # Adjust as per your classes

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(6,6))

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format='d')

    # Set title
    plt.title('Confusion Matrix')

    # Save the figure
    plt.savefig('confusion_matrix.png')
    plt.close()

    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout  # restore original


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    main()
