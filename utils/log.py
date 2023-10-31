import os
import numpy as np

EPS = 1e-8


def to_distmap(x, m=None, white_bkgd=True, min=None, max=None):
    x = x.cpu().numpy()
    m = m.cpu().numpy() if m is not None else np.ones_like(x)

    o = np.ones_like(x) if white_bkgd else np.zeros_like(x)
    xm = x[m[...,0]==1]
    min_val = xm.min() if min is None else min
    max_val = xm.max() if max is None else max
    o[m[...,0]==1] = (xm - min_val) / (max_val - min_val + EPS)
    return (255 * o).astype(np.uint8)


def to_normalmap(x, m=None, white_bkgd=True):
    x = x.cpu().numpy()
    m = m.cpu().numpy() if m is not None else np.ones_like(x)

    o = 255. * np.ones_like(x) if white_bkgd else np.zeros_like(x)
    xm = x[m[...,0]==1]
    o[m[...,0]==1] = 255 * (xm + 1.) / 2.
    return o.astype(np.uint8)


def to_colormap(x):
    x = x.cpu().numpy()
    o = (255 * np.clip(x, 0, 1)).astype(np.uint8)
    return o


def save_config(args):
    logdir = args.logdir
    expname = args.expname
    os.makedirs(os.path.join(logdir, expname), exist_ok=True)
    f = os.path.join(logdir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(logdir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())