import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='MATE')

    # dataset lists
    parser.add_argument('--rgb_list', default='ucf_x3d_train.txt',
                        help='list of rgb features (train)')
    parser.add_argument('--test_rgb_list', default='ucf_x3d_test.txt',
                        help='list of rgb features (test / val)')

    # k-fold
    parser.add_argument('--use_kfold', action='store_true',
                        help='enable stratified k-fold cross validation')
    parser.add_argument('--fold_id', type=int, default=0,
                        help='fold id')

    # experiment
    parser.add_argument('--comment', default='tiny')
    parser.add_argument('--dropout_rate', type=float, default=0.4)
    parser.add_argument('--attn_dropout_rate', type=float, default=0.15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=16)

    # model
    parser.add_argument('--model_name', default='model')
    parser.add_argument('--model_arch', default='fast')
    parser.add_argument('--pretrained_ckpt', default=None)

    # training
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=1)


    args = parser.parse_args()
    return args
