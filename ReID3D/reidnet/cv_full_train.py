# cv_full_train.py
import os
import copy
import itertools
from main import parser, run_training

def main():
    base_args = parser.parse_args() 

    # ---- Define search space ----
    lrs = [0.0005]#, 0.00002]
    batch_sizes = [base_args.class_per_batch]  # or [16, 32]
    optimizers = ['AdamW'] #W', 'Adam', 'sgd']

    results = []

    for lr, opt in itertools.product(lrs, optimizers):
        args = copy.deepcopy(base_args)

        # Override hyperparams for this run
        args.lr = lr
        args.optimizer = opt
       

        # Give each run its own checkpoint directory
        exp_name = f"lr{lr}_opt{opt}"

        ckpt_root = base_args.ckpt  # base folder from CLI
        seed = getattr(base_args, "cv_seed", None)
        tag = getattr(base_args, "run_tag", "")
        
        parts = [ckpt_root]
        if tag:
            parts.append(tag)
        if seed is not None:
            parts.append(f"seed_{seed}")
        parts.append(exp_name)
        
        args.ckpt = os.path.join(*parts)
        os.makedirs(args.ckpt, exist_ok=True)


        print("\n==============================")
        print(f"Running config: {exp_name}")
        print("==============================")

        metrics = run_training(args)
        print(
          f"[RESULT] {exp_name}: "
          f"best_top1={metrics['best_top1']:.4f}, "
          f"final_top1={metrics['final_top1']:.4f}, "
          f"mAP={metrics['final_map']:.4f}, "
          f"best_mAP={metrics['best_map']:.4f}"
        )


        results.append((exp_name, metrics))

    # Print summary at the end
    print("\n===== SUMMARY =====")
    for name, m in results:
        print(f"{name}: best_top1={m['best_top1']:.4f}, final_top1={m['final_top1']:.4f}, mAP={m['final_map']:.4f}")

    # Optionally: pick best config
    best = max(results, key=lambda x: x[1]['best_map'])
    print("\nBest config:", best[0], "with best_mAP=", best[1]['best_map'])


if __name__ == "__main__":
    main()
