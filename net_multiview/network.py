import os
import torch
import torch.nn as nn
import sys
sys.path.append('../')

from net_classifier.network import DualVisClassifier
from utils.layer import Siren
from utils.ray import get_rayparam_func

EPS = 1e-8


class RaySurfDNet(nn.Module):
    def __init__(self, D=8, W=256, input_ch=4, rgb_layer=0, w0_init=30.):
        super(RaySurfDNet, self).__init__()
        self.predict_rgb = True if rgb_layer > 0 else False
        n_ext = max(rgb_layer, 1)

        self.lf_encoder = nn.ModuleList([Siren(input_ch, W, w0=w0_init, is_first=True)] +
                                        [Siren(W, W) for i in range(1, D-n_ext)])
        self.dist_dense = nn.ModuleList([Siren(W, W) for i in range(n_ext-1)] +
                                        [Siren(W, 1, activation=nn.Identity())])
        if self.predict_rgb:
            self.color_dense = nn.ModuleList([Siren(W, W) for i in range(rgb_layer-1)] +
                                             [Siren(W, 3, activation=nn.Identity())])

    @staticmethod
    def get_features(layers, x):
        h = x
        for i, l in enumerate(layers):
            h = layers[i](h)
        return h

    def forward(self, x):
        outputs = {}
        feats = self.get_features(self.lf_encoder, x)
        outputs['dist'] = self.get_features(self.dist_dense, feats)

        if self.predict_rgb:
            outputs['rgb'] = self.get_features(self.color_dense, feats)
        return outputs


def create_net(args, scene_info, device):
    ray_fn, input_ch = get_rayparam_func(scene_info)

    # initialise classifier and load ckpt
    model_cls = DualVisClassifier(D=args.netdepth_cls, W=args.netwidth_cls,
                                  input_ch=input_ch, ext_layer=args.ext_layer_cls).to(device)
    if not args.eval_only:
        print('Reloading vis classifier from', args.ckpt_path_cls)
        cls_ckpt = torch.load(args.ckpt_path_cls)
        model_cls.load_state_dict(cls_ckpt['network_fn'])

    # initialise distance network for multiview optimization
    model = RaySurfDNet(D=args.netdepth, W=args.netwidth, input_ch=input_ch, rgb_layer=args.rgb_layer).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lrate, betas=(0.9, 0.999), capturable=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.N_iters, eta_min=args.lrate*0.01)


    ############# Load checkpoints #############
    ckpts = [os.path.join(args.logdir, args.expname, f) for f in sorted(os.listdir(
        os.path.join(args.logdir, args.expname))) if 'tar' in f]
    print('Found ckpts', ckpts)

    start = 0
    if len(ckpts) > 0:
        ckpt_path = ckpts[-1]
        print('Loading ckpt from:', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        model.load_state_dict(ckpt['network_fn'])
        optimizer.load_state_dict(ckpt['optimizer'])
        optimizer.param_groups[0]['capturable'] = True
        scheduler.load_state_dict(ckpt['scheduler'])
        scheduler.last_epoch = ckpt['global_step']

    return ray_fn, start, model, model_cls, optimizer, scheduler
