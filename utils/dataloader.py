import torch
import numpy as np
from utils.ray import get_rays_np
from data import load_blender, load_dmsr, load_scannet

EPS = 1e-8

dataloder_func = {
    "blender": load_blender.load_data,
    "dmsr": load_dmsr.load_data,
    "scannet": load_scannet.load_data
}


class Dataloader:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.N_rand = args.N_rand

        i_split, self.all_dists, self.all_images, masks, self.cam_poses, self.scene_info = \
            dataloder_func[args.dataset](args.datadir, args.trainskip, args.testskip)

        # restore scene info
        self.scene_info['sphere_radius'] = args.radius
        self.i_train, self.i_test = i_split
        print('TRAIN views are', self.i_train)
        print('TEST views are', self.i_test)

        # compute rays
        all_rays = []
        for i, pose in enumerate(self.cam_poses):
            rays_o, rays_d = get_rays_np(self.scene_info, pose)  # (H, W, 3), (H, W, 3), (H, W, 1)
            ray = np.concatenate([rays_o, rays_d], -1)
            all_rays.append(ray)
        all_rays = np.stack(all_rays, axis=0)

        self.rays, self.dists, self.masks, self.imgs = {}, {}, {}, {}
        for mode, split in zip(['train', 'test'], [self.i_train, self.i_test]):
            self.rays[mode] = np.reshape(all_rays[split], [-1, 6])
            self.dists[mode] = np.reshape(self.all_dists[split], [-1, 1])
            self.masks[mode] = np.reshape(masks[split], [-1, 1])
            if args.rgb_layer > 0:
                self.imgs[mode] = np.reshape(self.all_images[split], [-1, 3])

            # extract foreground rays for train/eval
            self.rays[mode+'_fg'] = self.rays[mode][self.masks[mode][:, 0]==1]
            self.dists[mode+'_fg'] = self.dists[mode][self.masks[mode][:, 0]==1]
            self.masks[mode+'_fg'] = self.masks[mode][self.masks[mode][:, 0]==1]
            if args.rgb_layer > 0:
                self.imgs[mode+'_fg'] = self.imgs[mode][self.masks[mode][:, 0]==1]


    def __call__(self, inds, mode):
        batch_rays = torch.Tensor(self.rays[mode][inds]).to(self.device)
        dists = torch.Tensor(self.dists[mode][inds]).to(self.device)
        masks = torch.Tensor(self.masks[mode][inds]).to(self.device)
        targ_dict = {'dist': dists, 'mask': masks}

        if self.args.rgb_layer > 0:
            images = torch.Tensor(self.imgs[mode][inds]).to(self.device)
            targ_dict['image'] = images

        return batch_rays, targ_dict