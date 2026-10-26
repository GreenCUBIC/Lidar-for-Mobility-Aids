# test_multiquery.py
import os
from main import parser, run_test_only, run_test_multiqueries

def main():
    # Parse arguments exactly like training / main.py
    args = parser.parse_args()
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    print("======================================")
    print("          SINGLE-SHOT TEST            ")
    print("======================================")
    single_stats = run_test_only(args)

    print("\n======================================")
    print("         MULTI-QUERY TEST (AVG)       ")
    print("======================================")
    multi_stats = run_test_multiqueries(args)

    # Final summary print
    print("\n========== FINAL SUMMARY ==========")
    print("Single-shot:")
    print(f"  top1 = {single_stats['top1']*100:.2f}%")
    print(f"  mAP  = {single_stats['mAP']*100:.2f}%")
    print("\nMulti-query (averaged over rounds):")
    print(f"  avg_top1 = {multi_stats['avg_top1']*100:.2f}%")
    print(f"  avg_mAP  = {multi_stats['avg_map']*100:.2f}%")
    print("===================================\n")

if __name__ == "__main__":
    main()
