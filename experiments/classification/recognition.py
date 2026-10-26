import sys
import os
import re
from collections import Counter, defaultdict
import random
import time

# UNCHANGED: Imports for data handling and torch modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, Sampler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, pairwise_distances
from torch.cuda.amp import autocast
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# UNCHANGED: Import your custom model and Point class
from custom_models import CustomPointCloudClassifier
from pointcept.models.utils.structure import Point

# 1-NN closed-set accuracy
def closed_set_knn_acc(emb, labels):
    e = emb.numpy()
    labels = np.asarray(labels)
    sims = e @ e.T
    np.fill_diagonal(sims, -np.inf)
    preds = labels[sims.argmax(1)]
    return accuracy_score(labels, preds)

# Leave-one-out 1-shot accuracy
def one_shot_leave_one_out_acc(emb, labels):
    e = emb.numpy()
    labels = np.asarray(labels)
    N, corr = len(labels), 0
    for i in range(N):
        q, qlab = e[i], labels[i]
        # build one-shot support
        seen = {}
        for j in range(N):
            if j==i: continue
            pid = labels[j]
            if pid not in seen:
                seen[pid] = j
        supp_idx = np.fromiter(seen.values(), int)
        sims = e[supp_idx] @ q
        pred = labels[supp_idx[sims.argmax()]]
        corr += (pred==qlab)
    return corr/N

# Recall@K (example: K=5)
def recall_at_k(emb, labels, K=5):
    e = emb.numpy()
    dist = pairwise_distances(e, metric="cosine")
    np.fill_diagonal(dist, np.inf)
    idx = dist.argsort(axis=1)[:, :K]
    labels = np.array(labels)
    hits = (labels[idx] == labels[:,None]).any(axis=1)
    return hits.mean()

# mAP
def mean_average_precision(emb, labels):
    e = emb.numpy()
    ids = np.array(labels)
    dist  = pairwise_distances(e, metric="cosine")
    np.fill_diagonal(dist, np.inf)
    order = dist.argsort(axis=1)
    aps = []
    for i in range(len(ids)):
        rel = (ids[order[i]]==ids[i])
        if not rel.any(): continue
        ranks = np.flatnonzero(rel)+1
        precisions = np.arange(1, len(ranks)+1)/ranks
        aps.append(precisions.mean())
    return float(np.mean(aps)) if aps else 0.0

# choose which one to monitor during training:
METRIC_TO_MONITOR = "map"  # options: "closed_knn", "one_shot", "recall5", "map"

def compute_metric(emb, ids):
    if METRIC_TO_MONITOR == "closed_knn":
        return closed_set_knn_acc(emb, ids)
    if METRIC_TO_MONITOR == "one_shot":
        return one_shot_leave_one_out_acc(emb, ids)
    if METRIC_TO_MONITOR == "recall5":
        return recall_at_k(emb, ids, K=5)
    if METRIC_TO_MONITOR == "map":
        return mean_average_precision(emb, ids)
    raise ValueError(f"Unknown metric {METRIC_TO_MONITOR}")

@torch.no_grad()
def extract_embeddings(model, loader, device):
    """
    Runs the model (in eval mode) over every sample in 'loader' and
    collects the L2-normalized embeddings plus their IDs.
    """
    model.eval()
    embs, ids = [], []
    for batch in loader:
        for i, sample in enumerate(batch["data_dict"]):
            # move tensors to device
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    sample[k] = v.to(device)
            # build the Point and forward
            point = Point(sample)
            point.serialization(order=model.backbone.order,
                                shuffle_orders=model.backbone.shuffle_orders)
            point.sparsify()
            out  = model(point)
            emb  = out["embedding"]            # your forward() must supply this
            embs.append(emb.cpu())
            ids.append(batch["id"][i])
    return torch.cat(embs, dim=0), ids

# UNCHANGED: Additional helper functions for augmentation (if needed) and setting the seed, etc.
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# UNCHANGED: Your dataset class that loads samples from folders
class PointCloudDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Folder containing ID -> class -> .csv.
            transform (callable, optional): Optional transform to apply to each sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []   # will hold (file_path, label, id_str, person_id)
        
        # map each id_str to a unique integer
        self.id_to_idx = {}
        next_idx = 0

        # 1) Collect .csv files under root_dir -> ID -> class
        id_dirs = sorted(d for d in os.listdir(root_dir)
                         if os.path.isdir(os.path.join(root_dir, d)))
        for id_str in id_dirs:
            # assign a numeric person_id if we haven't seen this id_str
            if id_str not in self.id_to_idx:
                self.id_to_idx[id_str] = next_idx
                next_idx += 1
            person_id = self.id_to_idx[id_str]

            id_path = os.path.join(root_dir, id_str)
            class_dirs = sorted(d for d in os.listdir(id_path)
                                if os.path.isdir(os.path.join(id_path, d)))
            for c_dir in class_dirs:
                label = int(c_dir)  # class label as before
                class_path = os.path.join(id_path, c_dir)
                for file_name in os.listdir(class_path):
                    if file_name.endswith(".csv"):
                        file_path = os.path.join(class_path, file_name)
                        # now store person_id as well
                        self.samples.append((file_path, label, id_str, person_id))

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        # unpack the four‐tuple
        file_path, label, id_str, person_id = self.samples[idx]
        
        # (rest of your loading logic is unchanged)
        file_name = os.path.basename(file_path)
        match = re.search(r'\(Frame (\d+)\)', file_name)
        frame_num = int(match.group(1)) if match else -1
        
        data = pd.read_csv(file_path, engine='python', on_bad_lines='warn')
        intensity = data.iloc[:, 0]
        coords = data.iloc[:, [7,8,9]].to_numpy(dtype=np.float32, copy=True)
        coords = np.ascontiguousarray(coords)
        
        intensity = pd.to_numeric(intensity, errors='coerce').values
        valid_idx = ~np.isnan(intensity)
        intensity = intensity[valid_idx] / 255.0
        coords = coords[valid_idx]
        
        intensity_tensor = torch.from_numpy(intensity).float().unsqueeze(1)
        coords_tensor    = torch.from_numpy(coords).float()
        coords_tensor    = coords_tensor - coords_tensor.mean(dim=0, keepdim=True)
        
        feat = torch.cat((coords_tensor, intensity_tensor), dim=1)
        N = coords_tensor.shape[0]
        batch = torch.zeros(N, dtype=torch.long)
        
        data_dict = {
            'coord': coords_tensor,
            'feat': feat,
            'batch': batch,
            'grid_size': 0.05
        }
        label_tensor    = torch.tensor(label, dtype=torch.long)
        person_tensor   = torch.tensor(person_id, dtype=torch.long)  # NEW

        sample = {
            'data_dict': data_dict,
            'label': label_tensor,        # if you still need class label
            'person_id': person_tensor,   # <— add this
            'id': id_str,
            'frame': frame_num
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


# UNCHANGED: Custom collate function remains the same
def custom_collate_fn(batch):
    data_dict_list = []
    label_list = []
    id_list = []
    frame_list = []  # collect frame numbers
    person_ids = []

    for sample in batch:
        data_dict_list.append(sample['data_dict'])
        label_list.append(sample['label'])
        id_list.append(sample['id'])
        frame_list.append(sample['frame'])
        person_ids.append(sample['person_id'])

    labels_tensor = torch.stack(label_list, dim=0)

    collated_batch = {
        'data_dict': data_dict_list,
        'label': labels_tensor,
        'person_id': torch.tensor(person_ids, dtype=torch.long),
        'id': id_list,
        'frame': frame_list  # include frames in the batch
    }
    return collated_batch

# UNCHANGED: EarlyStopping and other logging/training routines remain largely the same.
class EarlyStopping:
    """
    Early stops the training if validation accuracy doesn't improve
    after a given patience.
    """
    def __init__(self, patience=5, mode='max'):
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

# --- ADDED: Batch-Hard Triplet Loss Function ---
def batch_hard_triplet_loss(embeddings, labels, margin=1.0):
    """
    Compute the batch hard triplet loss.
    embeddings: Tensor of shape [batch_size, embedding_dim]
    labels: Tensor of shape [batch_size]
    """
    # Compute pairwise Euclidean distances
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2)

    # Create mask for positives and negatives
    labels = labels.unsqueeze(1)  # [batch_size, 1]
    mask_positive = (labels == labels.t())
    # Exclude self-comparisons (diagonal)
    mask_positive = mask_positive & ~torch.eye(mask_positive.size(0), dtype=torch.bool, device=mask_positive.device)
    
    mask_negative = (labels != labels.t())

    # For each anchor, get the hardest positive
    hardest_positive_dist, _ = (pairwise_dist * mask_positive.float()).max(dim=1)

    # For negatives, set non-negative distances to a large value and then take the minimum
    max_val = pairwise_dist.max() + 1
    masked_negative = pairwise_dist.clone()
    masked_negative[~mask_negative] = max_val
    hardest_negative_dist, _ = masked_negative.min(dim=1)
    
    losses = torch.clamp(hardest_positive_dist - hardest_negative_dist + margin, min=0.0)
    return losses.mean()

# --- ADDED: Custom PKSampler to ensure each batch has multiple examples per person ---
class PKSampler(Sampler):
    def __init__(self, data_source, P, K):
        """
        Args:
            data_source: Your dataset instance which has a .samples attribute.
            P (int): Number of unique person IDs per batch.
            K (int): Number of samples per person.
        """
        self.data_source = data_source
        self.P = P
        self.K = K
        self.person_to_indices = self._build_mapping()

    def _build_mapping(self):
        mapping = {}
        # Each sample in self.data_source.samples is (file_path, label, id_str)
        for idx, sample in enumerate(self.data_source.samples):
            person_id = sample[2]
            mapping.setdefault(person_id, []).append(idx)
        return mapping

    def __iter__(self):
        person_ids = list(self.person_to_indices.keys())
        random.shuffle(person_ids)
        batch = []
        for i in range(0, len(person_ids), self.P):
            selected_ids = person_ids[i:i+self.P]
            for pid in selected_ids:
                indices = self.person_to_indices[pid]
                if len(indices) < self.K:
                    sampled = random.choices(indices, k=self.K)
                else:
                    sampled = random.sample(indices, k=self.K)
                batch.extend(sampled)
            if len(batch) == self.P * self.K:
                yield from batch
                batch = []
        if batch:
            yield from batch

    def __len__(self):
        return len(self.data_source.samples)
'''
# --- CHANGED: Modify/Extend Model to output an embedding ---
def add_embedding_head(model, embedding_dim=128):
    """
    Assuming that the backbone outputs a feature of size 512 (adjust if needed),
    we add an embedding head that produces an embedding of the desired dimension.
    The head is appended to the model.
    """
    # Create a simple embedding head
    model.embedding_head = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, embedding_dim)
    )
    # Optionally, apply normalization in the forward pass:
    # We'll assume that later in the training loop, the output embedding is normalized.
    return model
'''
# --- CHANGED: New training routine using Triplet Loss ---
def train_model_triplet_loss(model, train_loader, val_loader, optimizer, device, num_epochs=10, margin=1.0, early_stopping_patience=5, eval_freq=5):
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    early_stopper = EarlyStopping(patience=early_stopping_patience, mode='min')
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        #SANITY CHECK
        if epoch == 0:
            # take the first batch from the train_loader
            first_train_batch = next(iter(train_loader))
            pids = first_train_batch['person_id'].to(device)
            print(">>> Training person_ids in first batch:", torch.unique(pids))

        # Training Phase
        model.train()
        running_loss = 0.0
        total_samples = 0

        for batch in train_loader:
            data_dict_list = batch['data_dict']
            labels = batch['person_id'].to(device)  # using label for metric learning (person identity)
            batch_size = len(data_dict_list)
            optimizer.zero_grad()          
            embeddings = []
            # Loop over samples in the batch
            for i in range(batch_size):
                sample_dict = data_dict_list[i]
                # Move tensors to the device
                for k, v in sample_dict.items():
                    if isinstance(v, torch.Tensor):
                        sample_dict[k] = v.to(device)
                point = Point(sample_dict)
                point.serialization(order=model.backbone.order, shuffle_orders=model.backbone.shuffle_orders)
                point.sparsify()
                # Forward pass: get both classification logits and embedding
                out = model(point)
                # Expect out to have 'embedding' key now
                emb = out.get('embedding')
                if emb is None:
                    raise ValueError("Model output does not contain 'embedding'. Did you attach the embedding head?")
                embeddings.append(emb)
            # Stack embeddings to [batch_size, embedding_dim]
            embeddings = torch.cat(embeddings, dim=0)
            # Normalize embeddings (L2-norm)
            embeddings = embeddings / embeddings.norm(p=2, dim=1, keepdim=True)
            
            loss = batch_hard_triplet_loss(embeddings, labels, margin=margin)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)
            optimizer.step()

            running_loss += loss.item() * batch_size
            total_samples += batch_size

        epoch_loss = running_loss / total_samples

        #SANITY CHECK
        if epoch == 0:
            first_val_batch = next(iter(val_loader))
            pids = first_val_batch['person_id'].to(device)
            print(">>> Validation person_ids in first batch:", torch.unique(pids))

        # Validation Phase (using the same triplet loss metric)
        model.eval()
        val_running_loss = 0.0
        val_total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                data_dict_list = batch['data_dict']
                labels = batch['person_id'].to(device)
                batch_size = len(data_dict_list)
                embeddings = []

                # ← SANITY CHECK: print the unique person IDs in the first val batch
                #if epoch == 0:
                #    print(">>> Validation batch person_ids:", torch.unique(labels))

                for i in range(batch_size):
                    sample_dict = data_dict_list[i]
                    for k, v in sample_dict.items():
                        if isinstance(v, torch.Tensor):
                            sample_dict[k] = v.to(device)
                    point = Point(sample_dict)
                    point.serialization(order=model.backbone.order, shuffle_orders=model.backbone.shuffle_orders)
                    point.sparsify()
                    out = model(point)
                    emb = out.get('embedding')
                    if emb is None:
                        raise ValueError("Model output does not contain 'embedding'.")
                    embeddings.append(emb)
                embeddings = torch.cat(embeddings, dim=0)
                embeddings = embeddings / embeddings.norm(p=2, dim=1, keepdim=True)
                val_loss = batch_hard_triplet_loss(embeddings, labels, margin=margin)
                val_running_loss += val_loss.item() * batch_size
                val_total_samples += batch_size

        val_epoch_loss = val_running_loss / val_total_samples
        print(f'Epoch {epoch+1}/{num_epochs} - Train Triplet Loss: {epoch_loss:.4f} | Val Triplet Loss: {val_epoch_loss:.4f}')
        
        # ——— CONDITIONAL METRIC EVAL —————————
        if ((epoch + 1) % eval_freq == 0) or (epoch + 1 == num_epochs):
            val_emb, val_ids = extract_embeddings(model, val_loader, device)
            val_score       = compute_metric(val_emb, val_ids)
            print(f'           Val {METRIC_TO_MONITOR}: {val_score*100:.2f}%')
            # optionally early-stop on this metric:
            # early_stopper(-val_score, model)

        # Early stopping based on validation loss (lower is better)
        early_stopper(val_epoch_loss, model, model_path=f'best_model_triplet.pth')
        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break
        
        scheduler.step()

    print('Triplet Loss training complete.')
    return model

# --- UNCHANGED: Other testing and analysis functions remain unchanged if you wish to combine classification evaluation.
# (If you plan to use the embedding for one-shot recognition, you might write a nearest-neighbor–based test function.)

# UNCHANGED: Tee and other helper functions
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
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"

def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # Remove 'module.' prefix
        else:
            new_state_dict[k] = v
    return new_state_dict
'''
def attach_metric_forward(model, l2_normalise=True):
    """
    Wrap the existing forward so that out['embedding'] is included.
    """
    orig_forward = model.forward            # keep original

    def metric_forward(point, *args, **kwargs):
        out = orig_forward(point, *args, **kwargs)   # e.g. {'cls_logits': …}

        # ------------- choose the backbone tensor -------------
        feats = out["feat"] if "feat" in out else out["backbone_feats"]
        # ^^^ adjust this line to grab the tensor coming FROM the backbone
        # ------------------------------------------------------

        emb  = model.embedding_head(feats)
        if l2_normalise:
            emb = nn.functional.normalize(emb, p=2, dim=1)

        out["embedding"] = emb
        return out

    model.forward = metric_forward          # monkey-patch
    return model
'''
# --- CHANGED: main() function adjusted to use PKSampler and triplet loss training ---
def main():
    # Seed
    train_start_time = time.time()
    set_seed(42)
    
    batch_size_value = 32  # Will be set as P*K in the PKSampler approach
    P = 8  # Number of distinct persons per batch
    K = 4  # Number of samples per person
    
    txt_string = f"training_log_triplet_bs{batch_size_value}.txt"
    tee = Tee(txt_string, "w")
    sys.stdout = tee

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    TARGET_CLASS = 0

    # Build Datasets (UNCHANGED)
    train_dataset = PointCloudDataset(root_dir='/data/seankirkby/data_subset/train')
    val_dataset   = PointCloudDataset(root_dir='/data/seankirkby/data_subset/val')
    test_dataset  = PointCloudDataset(root_dir='/data/seankirkby/data_subset/test')

    # 3) Filter each dataset *in place* so it only contains that one class
    def filter_to_target_class(dataset, target_label):
        # keep only those entries whose class label == TARGET_CLASS
        dataset.samples = [
            (fp, lbl, idd, pid)
            for (fp, lbl, idd, pid) in dataset.samples
            if lbl == target_label
        ]


    filter_to_target_class(train_dataset, TARGET_CLASS)
    filter_to_target_class(val_dataset,   TARGET_CLASS)
    filter_to_target_class(test_dataset,  TARGET_CLASS)

    print("Unique labels in train:", set(lbl for _,lbl,_,_ in train_dataset.samples))
    print("Unique labels in val:  ", set(lbl for _,lbl,_,_ in val_dataset.samples))

    # Create PKSampler for the training set (ADDED)
    #sampler = PKSampler(train_dataset, P=P, K=K)
    
    # 4) Create DataLoaders with standard shuffle (no PKSampler needed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_value,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_value,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_value,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn
    )
    
    # --- Model Setup ---
    model_cfg = {
        'num_classes': 5,
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

    # If there is a pre-trained model, load it (UNCHANGED)
    pretrained_model_path = 'model_best.pth'
    if os.path.exists(pretrained_model_path):
        state_dict = torch.load(pretrained_model_path, map_location=device)
        state_dict = remove_module_prefix(state_dict)
        model.load_state_dict(state_dict, strict=False)
        print("Loaded pre-trained weights from model_best.pth")
    else:
        print("Pre-trained model not found. Proceeding with randomly initialized weights.")

    # --- CHANGED: Attach an embedding head to the model for metric learning ---
    #model = add_embedding_head(model, embedding_dim=128)
    #model = attach_metric_forward(model) 
    
    # --- ADDED: Define optimizer for the entire model ---
    lr = 0.0002
    wd = 0.0002
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    # --- ADDED: Train the model using the triplet loss training routine ---
    num_epochs = 30
    margin = 1.0
    print("\n=== Training with Triplet Loss ===")
    model = train_model_triplet_loss(model, train_loader, val_loader, optimizer, device,
                                     num_epochs=num_epochs, margin=margin, early_stopping_patience=5)
    
    train_end_time = time.time()
    total_train_time = train_end_time - train_start_time
    print(f"TOTAL TIME TRAINING: {format_time(total_train_time)}")
    
    # UNCHANGED: Testing and analysis can be adapted similarly.
    # For example, you would create a testing routine that extracts embeddings and performs nearest neighbor matching.
    print("\n=== Testing the final model (embedding extraction) ===")
    # Here we simply run through the test set and collect embeddings.
    model.eval()
    all_ids = []
    all_embeddings = []
    with torch.no_grad():
        for batch in test_loader:
            data_dict_list = batch['data_dict']
            ids = batch['id']
            batch_embeddings = []
            for i in range(len(data_dict_list)):
                sample_dict = data_dict_list[i]
                for k, v in sample_dict.items():
                    if isinstance(v, torch.Tensor):
                        sample_dict[k] = v.to(device)
                point = Point(sample_dict)
                point.serialization(order=model.backbone.order, shuffle_orders=model.backbone.shuffle_orders)
                point.sparsify()
                out = model(point)
                emb = out.get('embedding')
                emb = emb / emb.norm(p=2, dim=1, keepdim=True)
                batch_embeddings.append(emb)
            batch_embeddings = torch.cat(batch_embeddings, dim=0)
            all_embeddings.append(batch_embeddings.cpu())
            all_ids.extend(ids)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    print("Extracted embeddings shape:", all_embeddings.shape)

    # ----- compute & print test metric -----
    test_score = compute_metric(all_embeddings, all_ids)
    print(f"\nTEST {METRIC_TO_MONITOR}: {test_score*100:.2f}%")

    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout  # restore original

if __name__ == "__main__":
    main()
