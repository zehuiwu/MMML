import argparse
from utils.audio_train import modelConfig, run

def main(args):
    run(modelConfig(learning_rate = args.lr,
                     dataset_name = args.dataset,
                     seed = args.seed,
                     feature = args.feature,
                     batch_size = args.batch_size,
                     early_stop = args.early_stop))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--dataset', type=str, default='sims', help='dataset name')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--feature', type=str, default='spec', help='feature type: spec, smile, or raw')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--early_stop', type=int, default=8, help='early stop')
    args = parser.parse_args()

    main(args)
