import os
import torch
import numpy as np
import imageio
import json
import trimesh

def load_data(basedir, trainskip=1, testskip=1):
    splits = ['train', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_dists, all_imgs, all_poses = [], [], []
    counts = [0]
    for s in splits:
        meta = metas[s]
        dists, imgs, poses = [], [], []
        skip = trainskip if s == 'train' else testskip

        for f_i, frame in enumerate(meta['frames']):
            if f_i % skip != 0:
                continue
            dists.append(np.load(os.path.join(basedir, frame['file_path'] + ".npy")))
            imgs.append(imageio.imread(os.path.join(basedir, frame['file_path'] + '.png')))
            poses.append(np.array(frame['transform_matrix']))

        dists = np.array(dists).astype(np.float32)
        imgs = (np.array(imgs) / 255.).astype(np.float32)[..., :3]
        poses = np.array(poses).astype(np.float32)

        all_dists.append(dists)
        all_imgs.append(imgs)
        all_poses.append(poses)

        counts.append(counts[-1] + dists.shape[0])

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(len(splits))]

    dists = np.concatenate(all_dists, 0)
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    masks = (dists > 0.).astype(np.float32)
    dists *= masks
    imgs = imgs * masks[..., None] + (1. - masks[..., None])

    # load scene mesh to normalize
    scene_name = basedir.split('/')[-1]
    scene = trimesh.load(os.path.join(basedir, scene_name+'.ply'))
    scene_center = (scene.bounds[1] + scene.bounds[0])/2.
    del scene

    H, W = dists[0].shape[:2]
    focal = meta['cam_info']['fx']
    K = np.array([
        [focal, 0, meta['cam_info']['cx']],
        [0, focal, meta['cam_info']['cy']],
        [0, 0, 1]
    ])
    scene_info = {
        'sphere_center': scene_center,
        'H': int(H),
        'W': int(W),
        'K': K,
        'focal': focal
    }

    return i_split, dists, imgs, masks, poses, scene_info
