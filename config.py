import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--eval_only", action='store_true',
                        help='only evaluation with pretrained model')

    # parameterization options
    parser.add_argument("--radius", type=float, default=1.5,
                        help='radius of sphere for distance field')

    # training options
    parser.add_argument("--N_rand", type=int, default=8192,
                        help='batch size')
    parser.add_argument("--N_iters", type=int, default=100000,
                        help='number of epochs')
    parser.add_argument("--lrate", type=float, default=1e-4,
                        help='learning rate')

    # classifier options
    parser.add_argument("--dist_thres", type=float, default=1e-2,
                        help='threshold to determine if the query point is occluded for the sampled view')
    parser.add_argument("--vis_thres", type=float, default=0.5,
                        help='threshold for binary classification')
    parser.add_argument("--netdepth_cls", type=int, default=8,
                        help='layers in visibilit classifier')
    parser.add_argument("--netwidth_cls", type=int, default=512,
                        help='channels per layer')
    parser.add_argument("--ext_layer_cls", type=int, default=1,
                        help='number of layers to extract individual features')
    parser.add_argument("--pos_weight", type=float, default=1.,
                        help='positive weight for cross-entropy loss')

    # multiview optimization options
    parser.add_argument("--N_views", type=int, default=20,
                        help='the number of reference views per ray')
    parser.add_argument("--w_rgb", type=float, default=1.,
                        help='weight of rgb loss')
    parser.add_argument("--ckpt_path_cls", type=str, default=None,
                        help='checkpoint path of classifier to reload')

    # ray-surface distance network
    parser.add_argument("--netdepth", type=int, default=13,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=1024,
                        help='channels per layer')
    parser.add_argument("--rgb_layer", type=int, default=0,
                        help='if true, network predicts radiance')
    parser.add_argument("--denoise", action='store_true',
                        help='if true, compute gradients to remove outliers')
    parser.add_argument("--grad_normal", action='store_true',
                        help='if true, use gradients to compute surface normal')
    parser.add_argument("--grad_clip", type=float, default=-1,
                        help='maximum clip value for grad norm')
    parser.add_argument("--outlier_thres", type=float, default=10.,
                        help='threshold to select outliers for minimizing the surface gradient')

    # dataset options
    parser.add_argument("--datadir", type=str, default='./datasets',
                        help='input data directory')
    parser.add_argument("--dataset", type=str, required=True,
                        help='the name of dataset for train/eval')
    parser.add_argument("--scene", type=str, required=True,
                        help='the name of scene for train/eval')
    parser.add_argument("--trainskip", type=int, default=1,
                        help='will load 1/N images from test/val sets')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets')
    parser.add_argument("--voxel_sz", type=float, default=0.005,
                        help='size of voxel for tsdf integration')
    parser.add_argument("--cd_sample", type=int, default=30000,
                        help='the number of sampling points to compute chamfer-distance')
    parser.add_argument("--continuous", action='store_true',
                        help='output continuous distance maps')

    # logging/saving options
    parser.add_argument("--logdir", type=str, default='./logs',
                        help='where to store ckpts and logs')
    parser.add_argument("--expname", type=str, default='',
                        help='experiment name')
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img", type=int, default=5000,
                        help='frequency of image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')

    return parser