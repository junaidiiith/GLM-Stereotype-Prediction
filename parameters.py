import argparse
import logging
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance", type=int, default=1)
    parser.add_argument("--mask_prob", type=float, default=0.2)
    parser.add_argument("--test_mask_prob", type=float, default=0.001)
    parser.add_argument("--min_stereotype_nodes", type=int, default=10)
    parser.add_argument("--exclude_limit", type=int, default=100)
    parser.add_argument("--model_name", type=str, default="bert-base-cased")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models_dir", type=str, default="data")

    parser.add_argument("--use_stereotypes", action='store_true')
    parser.add_argument("--use_rel_stereotypes", action='store_true')
    parser.add_argument("--use_context", action='store_true')
    
    # # training
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--lm_batch_size", type=int, default=64)
    parser.add_argument("--graph_batch_size", type=int, default=5)
    
    
    args = parser.parse_args()
    logging.info(args)
    return args


# if __name__ == "__main__":
#     args = parse_args()
