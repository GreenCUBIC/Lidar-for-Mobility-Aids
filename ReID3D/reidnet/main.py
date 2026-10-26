import parser
args = parser.parse_args()
import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
import math
import random
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
cudnn.benchmark=True
torch.multiprocessing.set_sharing_strategy('file_system')
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

import sys
sys.path.append('..')
from model import loss
from model import network
from util import utils

def np_cdist(feat1, feat2):
    """Cosine distance"""
    feat1_u = feat1 / np.linalg.norm(feat1, axis=1, keepdims=True) # n * d -> n
    feat2_u = feat2 / np.linalg.norm(feat2, axis=1, keepdims=True) # n * d -> n
    return -1 * np.dot(feat1_u, feat2_u.T)

def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def Video_acc(features, ids, q_idx):
    
    features = torch.tensor(features)
    q_idx = list(q_idx)
    g_idx = [i for i in range(len(ids)) if i not in q_idx]
    # import pdb;pdb.set_trace()
    fea_g = features[g_idx]
    fea_q = features[q_idx]
    ids_g = ids[g_idx]
    ids_q = ids[q_idx]
    acc_right = 0
    acc_right3 = 0
    
    right_ids = []
    ap = 0
    for query in q_idx:
        fea_dis = torch.tensor(np_cdist(fea_g, features[query].reshape(1,-1)).reshape(-1))
        min_idx = torch.argmin(fea_dis)
        a1, idx1 = torch.sort(fea_dis, descending=False)
        if ids_g[min_idx] == ids[query]:
            acc_right += 1
            right_ids.append(int(ids_g[min_idx]))
        if ids[query] in ids_g[idx1[:3]]:
            acc_right3 += 1
        
        good_index = [k for k in range(len(ids_g)) if ids_g[k] == ids[query]]
        ngood = len(good_index)
        mask = np.in1d(idx1.numpy(), good_index)
        rows_good = np.argwhere(mask==True)
        rows_good = rows_good.flatten()   
        for i in range(ngood):
            d_recall = 1.0/ngood
            precision = (i+1)*1.0/(rows_good[i]+1)
            ap = ap + d_recall*precision
        
    print('acc: ', acc_right/len(q_idx), acc_right3/len(q_idx), ap/len(q_idx), acc_right, len(q_idx))
    return acc_right/len(q_idx), acc_right3/len(q_idx),  ap/len(q_idx), acc_right, len(q_idx)    
      
def test_rrs(net, dataloader, args):

    net.eval()
    pbar = tqdm(total=len(dataloader), ncols=100, leave=True)
    pbar.set_description('Inference')

    gallery_features = []
    gallery_features_ske = []
    gallery_labels = []
    gallery_cams = []
    with torch.no_grad():
        for c, data in enumerate(dataloader):
            seqs = data[0].cuda()
            seqs = seqs.reshape((seqs.shape[0]//args.seq_len, args.seq_len, ) + seqs.shape[1:])
            label = data[1]
            cams = data[2]
            out = net(seqs)
            feat = out['val_bn']
            gallery_features.append(feat.cpu())
            gallery_labels.append(label)
            gallery_cams.append(cams)
            pbar.update(1)
    pbar.close()
    gallery_features = torch.cat(gallery_features, dim=0).numpy()
    gallery_labels = torch.cat(gallery_labels, dim=0).numpy()
    gallery_cams = torch.cat(gallery_cams, dim=0).numpy()
        
    acc = Video_acc(gallery_features, gallery_labels, dataloader.dataset.query_idx)
    net.train()
    return acc
    
def test_rrs_multiqueries(net, dataloader, args,
                          n_rounds=5,
                          queries_per_id=1):
    """
    Run evaluation multiple times with different random query sets and
    average the performance.

    - n_rounds: how many random query samplings.
    - queries_per_id: how many queries per identity per round.
    """
    net.eval()
    pbar = tqdm(total=len(dataloader), ncols=100, leave=True)
    pbar.set_description('Extract features (multi-query)')

    gallery_features = []
    gallery_labels = []
    gallery_cams = []
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            seqs = data[0].cuda()
            seqs = seqs.reshape(
                (seqs.shape[0] // args.seq_len, args.seq_len,) + seqs.shape[1:]
            )
            label = data[1]
            cams = data[2]
            out = net(seqs)
            feat = out['val_bn']
            gallery_features.append(feat.cpu())
            gallery_labels.append(label)
            gallery_cams.append(cams)
            pbar.update(1)
    pbar.close()

    gallery_features = torch.cat(gallery_features, dim=0).numpy()
    gallery_labels = torch.cat(gallery_labels, dim=0).numpy()
    gallery_cams = torch.cat(gallery_cams, dim=0).numpy()

    # Group indices by person ID
    id_to_indices = {}
    for idx, pid in enumerate(gallery_labels):
        pid = int(pid)
        id_to_indices.setdefault(pid, []).append(idx)

    n_id = len(id_to_indices)
    print(f"[MULTI-QUERY] Found {n_id} unique IDs in test set.")

    # Allow CLI override
    n_rounds = getattr(args, "n_query_rounds", n_rounds)
    queries_per_id = getattr(args, "queries_per_id", queries_per_id)

    all_top1 = []
    all_top3 = []
    all_map = []
    all_right = []
    all_total = []

    for r in range(n_rounds):
        base_seed = getattr(args, "cv_seed", 0)
        if base_seed is None:
          base_seed = 0
        rnd = random.Random(base_seed + r)

        q_idx = []
        for pid, idxs in id_to_indices.items():
            idxs_local = idxs[:]
            rnd.shuffle(idxs_local)
            take = min(len(idxs_local), int(queries_per_id))
            q_idx.extend(idxs_local[:take])

        q_idx = sorted(set(q_idx))
        print(f"[MULTI-QUERY] Round {r + 1}/{n_rounds}: #queries={len(q_idx)}")

        acc, acc3, m_ap, num_right, num_all = Video_acc(
            gallery_features, gallery_labels, q_idx
        )

        all_top1.append(acc)
        all_top3.append(acc3)
        all_map.append(m_ap)
        all_right.append(num_right)
        all_total.append(num_all)

    avg_top1 = float(np.mean(all_top1)) if all_top1 else 0.0
    avg_top3 = float(np.mean(all_top3)) if all_top3 else 0.0
    avg_map = float(np.mean(all_map)) if all_map else 0.0

    print("\n===== MULTI-QUERY AVERAGE RESULTS =====")
    print(f"  rounds        : {n_rounds}")
    print(f"  queries/id    : {queries_per_id}")
    print(f"  avg top-1     : {avg_top1 * 100:.2f}%")
    print(f"  avg top-3     : {avg_top3 * 100:.2f}%")
    print(f"  avg mAP       : {avg_map * 100:.2f}%")
    print("=======================================\n")

    net.train()
    return {
        "avg_top1": avg_top1,
        "avg_top3": avg_top3,
        "avg_map": avg_map,
        "round_top1": [float(x) for x in all_top1],
        "round_top3": [float(x) for x in all_top3],
        "round_map": [float(x) for x in all_map],
        "round_num_right": [int(x) for x in all_right],
        "round_num_all": [int(x) for x in all_total],
    }


def run_training(args):
    torch.manual_seed(4)
    np.random.seed(4)
    random.seed(4)
    num = 4
    torch.set_num_threads(num)

    print('\nDataloading starts !!')

    train_dataloader = utils.Get_Video_train_DataLoader(
        args.train_txt, args.train_info, shuffle=True, num_workers=args.num_workers,
        seq_len=args.seq_len, track_per_class=args.track_per_class, class_per_batch=args.class_per_batch)

    test_rrs_dataloader = utils.Get_Video_test_rrs_DataLoader(
        args.test_txt, args.test_info, args.query_info, batch_size=args.test_batch,
        shuffle=False, num_workers=args.num_workers, seq_len=args.seq_len, distractor=True)

    print('Dataloading ends !!\n')

    num_class = train_dataloader.dataset.n_id
    net = nn.DataParallel(
    network.reid3d(args.feat_dim, num_class=num_class, stride=args.stride).cuda())

    # ---- Load pretrained weights if requested ----
    if getattr(args, "load_ckpt", None):
        print(f"[INFO] Loading checkpoint for finetuning: {args.load_ckpt}")
        ckpt = torch.load(args.load_ckpt, map_location="cpu")

        # Handle checkpoints that wrap params in "state_dict"
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        model_state = net.state_dict()
        filtered_state = {}

        for k, v in ckpt.items():
            if k not in model_state:
                # Key doesn't exist in current model (new arch / renamed)
                print(f"[CKPT] Skip missing key: {k}")
                continue

            if model_state[k].shape != v.shape:
                # Shape mismatch (e.g., old classifier with 222 IDs vs new 28 IDs)
                print(f"[CKPT] Skip shape-mismatch key: {k} "
                      f"{tuple(v.shape)} -> {tuple(model_state[k].shape)}")
                continue

            filtered_state[k] = v

        missing, unexpected = net.load_state_dict(filtered_state, strict=False)
        print(f"[CKPT] Loaded {len(filtered_state)} keys "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")


    os.system('mkdir -p %s' % (args.ckpt))

    # optimizer
    # optimizer
    opt_name = str(getattr(args, "optimizer", "adamw")).lower()
    
    if opt_name == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    elif opt_name == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-5)
    else:  # 'adamw' default
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-5)


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs, eta_min=args.lr * 0.01, last_epoch=-1, verbose=False)

    best_cmc = 0.0
    best_map = 0.0
    crit = loss.Loss().cuda()

    for epoch in range(0, args.n_epochs):

        ############################### Training ###############################
        pbar = tqdm(total=len(train_dataloader), ncols=100, leave=True)
        pbar.set_description('Epoch %03d' % epoch)

        loss_all = 0
        loss_id = 0
        loss_trip = 0
        loss_id_final = 0
        loss_trip_final = 0

        for batch_idx, data in enumerate(train_dataloader):
            seqs, labels = data
            seqs = seqs.cuda()
            labels = labels.cuda()

            out = net(seqs)
            loss_out = crit(out, labels)

            total_loss = 0
            total_loss += loss_out['track_id']
            total_loss += loss_out['trip']

            loss_all += float(total_loss)
            loss_id += float(loss_out['track_id'])
            loss_trip += float(loss_out['trip'])

            loss_id_final += float(loss_out['track_id_final'])
            loss_trip_final += float(loss_out['trip_final'])

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            pbar.update(1)

        pbar.close()
        print('lr: ', optimizer.state_dict()['param_groups'][0]['lr'], 'total_loss: ', loss_all,
              'track_id_loss: ', loss_id, 'trip_loss: ', loss_trip,
              'track_id_final: ', loss_id_final, 'trip_final: ', loss_trip_final)

        # Save last
        torch.save(net.state_dict(), os.path.join(args.ckpt, 'ckpt_last.pth'))

        if args.lr_step_size != 0:
            scheduler.step()

        # log train losses
        with open(os.path.join(args.ckpt, args.log_path), 'a') as f:
            f.write('lr: %f total loss: %.4f track_id loss: %.4f trip loss: %.4f '
                    'track_id final: %.4f trip final: %.4f \n' %
                    (optimizer.state_dict()['param_groups'][0]['lr'],
                     loss_all, loss_id, loss_trip, loss_id_final, loss_trip_final))
        '''
        ############################### Validation ###############################
        if (epoch + 1) % args.eval_freq == 0:
            acc, acc3, m_ap, num_right, num_all = test_rrs(net, test_rrs_dataloader, args)

            with open(os.path.join(args.ckpt, args.log_path), 'a') as f:
                f.write('[Epoch %03d] top1: %.1f top3: %.1f map: %.1f num_right: %d num_all: %d \n' %
                        (epoch, acc * 100, acc3 * 100, m_ap * 100, num_right, num_all))

            if acc >= best_cmc:
                torch.save(net.state_dict(), os.path.join(args.ckpt, 'ckpt_best.pth'))
                best_cmc = acc
        '''
        
        ############################### Validation ###############################
        if (epoch + 1) % args.eval_freq == 0:
            acc, acc3, m_ap, num_right, num_all = test_rrs(net, test_rrs_dataloader, args)
        
            with open(os.path.join(args.ckpt, args.log_path), 'a') as f:
                f.write('[Epoch %03d] top1: %.1f top3: %.1f map: %.1f num_right: %d num_all: %d \n' %
                        (epoch, acc * 100, acc3 * 100, m_ap * 100, num_right, num_all))
        
            # Select best by mAP
            if m_ap >= best_map:
                torch.save(net.state_dict(), os.path.join(args.ckpt, 'ckpt_best.pth'))
                best_map = m_ap
                best_cmc = acc

        
    # Final test at the end
    print('----Test---- \n')
    acc, acc3, m_ap, num_right, num_all = test_rrs(net, test_rrs_dataloader, args)

    with open(os.path.join(args.ckpt, args.log_path), 'a') as f:
      f.write('[Test] top1: %.1f top3: %.1f map: %.1f num_right: %d num_all: %d \n' %
                (acc * 100, acc3 * 100, m_ap * 100, num_right, num_all))

    with open(os.path.join(args.ckpt, args.log_path), 'a') as f:
      f.write('best map: %.1f \n' % (best_map * 100))
      f.write('best top1: %.1f \n' % (best_cmc * 100))

    #print('best_acc: ', best_cmc)
    print('best_map: ', best_map)

    # ---- Write best-metrics summary file ----
    try:
        summary_path = os.path.join(args.ckpt, "best_metrics.txt")
        with open(summary_path, "w") as f:
            f.write(f"best_mAP: {best_map:.6f}\n")
            f.write(f"best_top1: {best_cmc:.6f}\n")
            f.write(f"final_mAP: {m_ap:.6f}\n")
            f.write(f"final_top1: {acc:.6f}\n")
        print(f"[INFO] Wrote {summary_path}")
    except Exception as e:
        print(f"[WARN] Could not write best_metrics.txt: {e}")

    # return something for hyperparam search
    return {
      "best_top1": float(best_cmc),   # <-- now exists for full_train.py
      "best_map": float(best_map),    # <-- still your selection criterion
      "final_top1": float(acc),
      "final_map": float(m_ap)
    }

def run_test_only(args):
    """
    Test the model once using the dataset's query_idx (single evaluation).
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    torch.manual_seed(4)
    np.random.seed(4)
    random.seed(4)
    torch.set_num_threads(4)

    print('\n[TEST-ONLY] Dataloading starts !!')

    train_dataloader = utils.Get_Video_train_DataLoader(
        args.train_txt, args.train_info, shuffle=False, num_workers=args.num_workers,
        seq_len=args.seq_len, track_per_class=args.track_per_class, class_per_batch=args.class_per_batch)

    test_rrs_dataloader = utils.Get_Video_test_rrs_DataLoader(
        args.test_txt, args.test_info, args.query_info, batch_size=args.test_batch,
        shuffle=False, num_workers=args.num_workers, seq_len=args.seq_len, distractor=True)

    print('[TEST-ONLY] Dataloading ends !!\n')

    num_class = train_dataloader.dataset.n_id

    net = nn.DataParallel(
        network.reid3d(args.feat_dim, num_class=num_class, stride=args.stride).cuda())

    # Load checkpoint
    if getattr(args, "load_ckpt", None):
        print(f"[TEST-ONLY] Loading checkpoint: {args.load_ckpt}")
        ckpt = torch.load(args.load_ckpt, map_location="cpu")

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        model_state = net.state_dict()
        filtered_state = {}
        for k, v in ckpt.items():
            if k not in model_state:
                print(f"[CKPT] (test-only) Skip missing key: {k}")
                continue
            if model_state[k].shape != v.shape:
                print(f"[CKPT] (test-only) Skip shape-mismatch key: {k} "
                      f"{tuple(v.shape)} -> {tuple(model_state[k].shape)}")
                continue
            filtered_state[k] = v

        missing, unexpected = net.load_state_dict(filtered_state, strict=False)
        print(f"[CKPT] (test-only) Loaded {len(filtered_state)} keys "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        print("[WARN] [TEST-ONLY] --load_ckpt not set; testing random weights.")

    print("\n[TEST-ONLY] Running evaluation once with dataset.query_idx\n")
    top1, top3, m_ap, num_right, num_all = test_rrs(net, test_rrs_dataloader, args)

    print(f"[TEST-ONLY RESULT] top1={top1*100:.2f}%, top3={top3*100:.2f}%, "
          f"mAP={m_ap*100:.2f}%, num_right={num_right}, num_all={num_all}")

    # Optional: write a small summary file in args.ckpt
    if getattr(args, "ckpt", None):
        try:
            summary_path = os.path.join(args.ckpt, "test_only_summary.txt")
            os.makedirs(args.ckpt, exist_ok=True)
            with open(summary_path, "w") as f:
                f.write(f"top1: {top1:.6f}\n")
                f.write(f"top3: {top3:.6f}\n")
                f.write(f"mAP: {m_ap:.6f}\n")
                f.write(f"num_right: {num_right}\n")
                f.write(f"num_all: {num_all}\n")
            print(f"[INFO] Wrote {summary_path}")
        except Exception as e:
            print(f"[WARN] Could not write test_only_summary.txt: {e}")

    return {
        "top1": float(top1),
        "top3": float(top3),
        "mAP": float(m_ap),
        "num_right": int(num_right),
        "num_all": int(num_all),
    }

def run_test_multiqueries(args):
    """
    Test-only entry that runs multiple random query samplings and averages metrics.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    torch.manual_seed(4)
    np.random.seed(4)
    random.seed(4)
    torch.set_num_threads(4)

    print('\n[TEST MULTI-QUERY] Dataloading starts !!')

    train_dataloader = utils.Get_Video_train_DataLoader(
        args.train_txt, args.train_info, shuffle=False, num_workers=args.num_workers,
        seq_len=args.seq_len, track_per_class=args.track_per_class, class_per_batch=args.class_per_batch)

    test_rrs_dataloader = utils.Get_Video_test_rrs_DataLoader(
        args.test_txt, args.test_info, args.query_info, batch_size=args.test_batch,
        shuffle=False, num_workers=args.num_workers, seq_len=args.seq_len, distractor=True)

    print('[TEST MULTI-QUERY] Dataloading ends !!\n')

    num_class = train_dataloader.dataset.n_id

    net = nn.DataParallel(
        network.reid3d(args.feat_dim, num_class=num_class, stride=args.stride).cuda())

    # Load checkpoint
    if getattr(args, "load_ckpt", None):
        print(f"[TEST MULTI-QUERY] Loading checkpoint: {args.load_ckpt}")
        ckpt = torch.load(args.load_ckpt, map_location="cpu")

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        model_state = net.state_dict()
        filtered_state = {}
        for k, v in ckpt.items():
            if k not in model_state:
                print(f"[CKPT] (test-mq) Skip missing key: {k}")
                continue
            if model_state[k].shape != v.shape:
                print(f"[CKPT] (test-mq) Skip shape-mismatch key: {k} "
                      f"{tuple(v.shape)} -> {tuple(model_state[k].shape)}")
                continue
            filtered_state[k] = v

        missing, unexpected = net.load_state_dict(filtered_state, strict=False)
        print(f"[CKPT] (test-mq) Loaded {len(filtered_state)} keys "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        print("[WARN] [TEST MULTI-QUERY] --load_ckpt not set; testing random weights.")

    # Allow CLI override
    n_rounds = getattr(args, "n_query_rounds", 5)
    queries_per_id = getattr(args, "queries_per_id", 1)

    stats = test_rrs_multiqueries(
        net, test_rrs_dataloader, args,
        n_rounds=n_rounds,
        queries_per_id=queries_per_id
    )

    # Optional: write summary
    if getattr(args, "ckpt", None):
        try:
            summary_path = os.path.join(args.ckpt, "multiquery_test_summary.txt")
            os.makedirs(args.ckpt, exist_ok=True)
            with open(summary_path, "w") as f:
                f.write(f"n_rounds: {n_rounds}\n")
                f.write(f"queries_per_id: {queries_per_id}\n")
                f.write(f"avg_top1: {stats['avg_top1']:.6f}\n")
                f.write(f"avg_top3: {stats['avg_top3']:.6f}\n")
                f.write(f"avg_mAP: {stats['avg_map']:.6f}\n")
            print(f"[INFO] Wrote {summary_path}")
        except Exception as e:
            print(f"[WARN] Could not write multiquery_test_summary.txt: {e}")

    return stats

if __name__ == '__main__':
    args = parser.parse_args()
    run_training(args)