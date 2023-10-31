import os
import numpy as np
import imageio
import json
import trimesh

def load_data(basedir, trainskip=1, testskip=1):
    splits = ['train', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, s, 'transforms.json'), 'r') as fp:
            metas[s] = json.load(fp)

    all_dists, all_masks, all_imgs, all_poses = [], [], [], []
    counts = [0]
    for s in splits:
        meta = metas[s]
        dists, masks, imgs, poses = [], [], [], []
        skip = trainskip if s == 'train' else testskip

        for f_i, frame in enumerate(meta['frames']):
            if f_i % skip != 0:
                continue
            fname = frame['file_path'].split('/')[-1]
            idx = fname.split('_')[-1]
            dists.append(np.load(os.path.join(basedir, s, 'distance', f'd_{idx}.npy')))
            masks.append(np.load(os.path.join(basedir, s, 'mask', f'{idx}.npy')))
            imgs.append(imageio.imread(os.path.join(basedir, s, 'rgbs', f'r_{idx}.png')))
            poses.append(np.array(frame['transform_matrix']))

        dists = np.array(dists).astype(np.float32)
        masks = np.array(masks).astype(np.float32)
        imgs = (np.array(imgs) / 255.).astype(np.float32)[..., :3]
        poses = np.array(poses).astype(np.float32)

        all_dists.append(dists)
        all_masks.append(masks)
        all_imgs.append(imgs)
        all_poses.append(poses)

        counts.append(counts[-1] + dists.shape[0])

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(len(splits))]

    dists = np.concatenate(all_dists, 0)
    masks = np.concatenate(all_masks, 0)
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    dists *= masks
    imgs = imgs * masks[..., None] + (1. - masks[..., None])

    # load scene mesh to normalize
    # scene_name = basedir.split('/')[-1]
    # scene = trimesh.load(os.path.join(basedir, scene_name + '.ply'))
    # scene_center = (scene.bounds[1] + scene.bounds[0]) / 2.

    H, W = dists[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    K = np.array([
        [focal, 0, 0.5 * W],
        [0, -focal, 0.5 * H],
        [0, 0, -1]
    ])
    scene_info = {
        'sphere_center': [0., 0., 0.],
        'H': int(H),
        'W': int(W),
        'K': K,
        'focal': focal
    }

    return i_split, dists, imgs, masks, poses, scene_info
