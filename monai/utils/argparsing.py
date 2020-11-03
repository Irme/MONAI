from argparse import ArgumentParser

def process(args=None):
    parser = ArgumentParser(description="Settings for network")
    parser.add_argument('-loss_type',type=str, default='dice')
    parser.add_argument('-patch_size', type=int, default=96)
    parser.add_argument('-batch_size', type=int, default=4)
    parser.add_argument('-border', type=int, default=42)
    parser.add_argument('-save_dir',type=str, default='/mnt/data/igroothuis/irme_ptproto/models/youshouldvesetthis')
    parser.add_argument('-sampling',type=str, default='uniform')
    parser.add_argument('-max_iter',type=int, default=10000)
    parser.add_argument('-eval_every',type=int, default=100)
    parser.add_argument('-weight_decay', type=float, default=0.0)
    parser.add_argument('-lr', type=float, default=1e3)
    parser.add_argument('-dropout', type=float, default=0.0)
    arguments = parser.parse_args(args)
    return arguments