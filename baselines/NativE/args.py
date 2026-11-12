import argparse


def get_args():
    arg = argparse.ArgumentParser()
    arg.add_argument('-dataset', type=str, default='FB15K')
    arg.add_argument('-batch_size', type=int, default=1024)
    arg.add_argument('-margin', type=float, default=6.0)
    arg.add_argument('-dim', type=int, default=128)
    arg.add_argument('-epoch', type=int, default=1000)
    arg.add_argument('-save', type=str)
    arg.add_argument('-img_dim', type=int, default=4096)
    arg.add_argument('-neg_num', type=int, default=1)
    arg.add_argument('-learning_rate', type=float, default=0.001)
    arg.add_argument('-lrg', type=float, default=0.001)
    arg.add_argument('-lrd', type=float, default=0.001)
    arg.add_argument('-adv_temp', type=float, default=2.0)
    arg.add_argument('-visual', type=str, default='random')
    arg.add_argument('-seed', type=int, default=42)
    arg.add_argument('-rank_file', type=str, default='', help='write per-triple filtered ranks (CSV) for scripts/compute_metrics.py')
    # [Geometry-MMKG] checkpoint selection (mmkgc/config/ValidSelector.py)
    arg.add_argument('-select_on', type=str, default='valid', choices=['valid', 'original'],
                     help='valid: keep the best validation-MRR checkpoint, test once; original: upstream behaviour')
    arg.add_argument('-valid_steps', type=int, default=10, help='evaluate on valid every N epochs')
    arg.add_argument('-patience', type=int, default=0, help='early stop after N evaluations without improvement (0 = off)')
    arg.add_argument('-valid_max', type=int, default=0, help='evaluate on a seeded random subset of N valid triples (0 = all)')
    arg.add_argument('-missing_rate', type=float, default=0.8)
    arg.add_argument('-postfix', type=str, default='')
    arg.add_argument('-con_temp', type=float, default=0)
    arg.add_argument('-lamda', type=float, default=0)
    arg.add_argument('-mu', type=float, default=0)
    arg.add_argument('-adv_num', type=int, default=1)
    arg.add_argument('-disen_weight', type=float, default=0.01)
    arg.add_argument('-miss_type', type=str, default=None)
    arg.add_argument('-miss_prop', type=float, default=None)
    return arg.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)
