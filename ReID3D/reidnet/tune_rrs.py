# tune_rrs.py
import os
import copy
import itertools
from main import parser, run_training  # adjust filename if different

def main():
    base_args = parser.parse_args() 

    # ---- Define search space ----
    lrs = [0.00002, 0.000005, 0.00001, 0.00002, 0.00005, 0.0005]
    batch_sizes = [base_args.class_per_batch]  # or [16, 32]
    optimizers = ['AdamW', 'Adam', 'sgd']

    results = []

    for lr, opt in itertools.product(lrs, optimizers):
        args = copy.deepcopy(base_args)

        # Override hyperparams for this run
        args.lr = lr
        args.optimizer = opt
       

        # Give each run its own checkpoint directory
        exp_name = f"lr{lr}_opt{opt}"
        args.ckpt = os.path.join("400_650_checkpoints_tune", exp_name)
        os.makedirs(args.ckpt, exist_ok=True)

        print("\n==============================")
        print(f"Running config: {exp_name}")
        print("==============================")

        metrics = run_training(args)
        print(f"[RESULT] {exp_name}: best_top1={metrics['best_top1']:.4f}, "
              f"final_top1={metrics['final_top1']:.4f}, mAP={metrics['final_map']:.4f}")

        results.append((exp_name, metrics))

    # Print summary at the end
    print("\n===== SUMMARY =====")
    for name, m in results:
        print(f"{name}: best_top1={m['best_top1']:.4f}, final_top1={m['final_top1']:.4f}, mAP={m['final_map']:.4f}")

    # Optionally: pick best config
    best = max(results, key=lambda x: x[1]['best_top1'])
    print("\nBest config:", best[0], "with best_top1=", best[1]['best_top1'])

if __name__ == "__main__":
    main()
