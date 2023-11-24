import os
import torch
import numpy as np
from tqdm import trange
from config import config_parser
import imageio
import trimesh
import open3d as o3d
from open3d import pipelines
import wandb
from wandb import AlertLevel

from utils import log
from utils.math import convert_d
from utils.dataloader import Dataloader
from utils.ray import get_ray_param
from net_multiview.network import create_net
from net_multiview.sampler import get_multiview_rays
from utils.math import get_surface_gradient, get_surface_normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
np.random.seed(0)

from torchmetrics.functional import peak_signal_noise_ratio as PSNR
from torchmetrics.functional import structural_similarity_index_measure as SSIM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
LPIPS = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)
from chamfer_distance import ChamferDistance
CD = ChamferDistance().to(device)

def train(args):
    # Load dataset
    dataloader = Dataloader(args, device)

    # Create rayparam function and network
    ray_fn, global_step, model, model_cls, optimizer, scheduler = create_net(args, dataloader.scene_info, device)

    # Create experiment logger
    wandb.init(project="RayDF-RaySurfDNet")
    wandb.run.name = args.expname
    wandb.watch(model, log="all")

    start = global_step
    train_num = len(dataloader.dists['train_fg'])
    inds = np.random.permutation(train_num)
    step_batch = train_num // args.N_rand

    for i in trange(start, args.N_iters):
        optimizer.zero_grad()
        j = i % step_batch
        ep = i // step_batch
        # re-random train indices after one epoch
        if j == 0 and i != start:
            inds = np.random.permutation(train_num)

        # =================== Query Rays ========================
        # Random rays from all images
        train_i = inds[j * args.N_rand: (j+1) * args.N_rand]

        # load query rays
        batch_rays, target_dict = dataloader(inds=train_i, mode='train_fg')
        batch_inputs, d0, _ = get_ray_param(ray_fn, batch_rays)

        # normalize query gt distance and query surface point
        target_dict['dist_norm'] = (target_dict['dist'] - d0) / (args.radius * 2.)
        batch_pts = batch_rays[..., :3] + target_dict['dist'] * batch_rays[..., 3:]
        for c in range(batch_pts.shape[-1]):
            batch_pts[..., c] -= dataloader.scene_info['sphere_center'][c]
        target_dict['pts_norm'] = batch_pts / dataloader.scene_info['sphere_radius']

        # ================= Multiview Rays =====================
        # Sample multiview rays and get their ray parameters
        mv_rays, mv_targets = get_multiview_rays(args, query_rays=batch_rays, query_gts=target_dict)
        mv_inputs, mv_d0, _ = get_ray_param(ray_fn, mv_rays)
        mv_targets['dist_norm'] = (mv_targets['dist'] - mv_d0) / (args.radius * 2.)

        # Compute visibility
        with torch.no_grad():
            cls_inputs = [torch.tile(batch_inputs[:, None], (1, args.N_views, 1)).reshape(-1, batch_inputs.shape[-1]),
                          mv_inputs,
                          torch.tile(target_dict['pts_norm'][:, None], (1, args.N_views, 1)).reshape(-1, 3)]
            vis = model_cls(cls_inputs)
            mv_targets['vis_score'] = torch.sigmoid(vis).reshape(args.N_rand, args.N_views)
            reweigh = 0.5
            mv_targets['vis_score'] = mv_targets['vis_score'] ** reweigh / (mv_targets['vis_score'] ** reweigh +
                                       (1. - mv_targets['vis_score']) ** reweigh)

        # Multiview forward
        mv_batch_inputs = torch.cat([batch_inputs, mv_inputs], dim=0)
        mv_outputs = model(mv_batch_inputs)

        # ================= Optimization =====================
        loss_d_query = torch.abs(mv_outputs['dist'][:args.N_rand] - target_dict['dist_norm'])[:, 0]
        loss_d_mv = torch.abs(mv_outputs['dist'][args.N_rand:] - mv_targets['dist_norm']).reshape(args.N_rand, args.N_views)
        loss_d = loss_d_query + (mv_targets['vis_score'] * loss_d_mv).sum(-1)
        loss_d = (loss_d / (1. + mv_targets['vis_score'].sum(-1))).mean()

        loss_rgb = torch.tensor(0.).to(device)
        if 'rgb' in mv_outputs:
            mv_outputs['rgb_pred'] = torch.sigmoid(mv_outputs['rgb'])
            loss_rgb_query = ((mv_outputs['rgb_pred'][:args.N_rand] - target_dict['image'])**2).mean(-1)
            loss_rgb_mv = ((mv_outputs['rgb_pred'][args.N_rand:] - mv_targets['image'])**2).reshape(args.N_rand, args.N_views, 3).mean(-1)
            loss_rgb = loss_rgb_query + (mv_targets['vis_score'] * loss_rgb_mv).sum(-1)
            loss_rgb = (loss_rgb / (1. + mv_targets['vis_score'].sum(-1))).mean()

        loss = loss_d + args.w_rgb * loss_rgb
        loss.backward()

        if args.grad_clip > 0.:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()
        new_lrate = optimizer.param_groups[0]['lr']
        scheduler.step()

        # ================= Logging ==========================
        if i % args.i_print == 0 and i != start:
            wandb.log({
                'train/_ep': ep,
                'train/_lr': new_lrate,
                'train/loss': loss.item(),
                'train/loss_d': loss_d.item(),
                'train/loss_rgb': loss_rgb.item()
            })

        # ================= Evaluation =====================
        if i % args.i_img == 0 and i != start:
            torch.cuda.empty_cache()
            eval(args, dataloader, ray_fn, model, mode='train')
            eval(args, dataloader, ray_fn, model, mode='test')

        # Save checkpoints
        if (i % args.i_weights == 0 and i != start) or (i + 1) == args.N_iters:
            path = os.path.join(args.logdir, args.expname, '{:07d}.tar'.format(i))
            ckpt_dict = {
                'global_step': global_step,
                'network_fn': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(ckpt_dict, path)
            print('Saved checkpoints at', path)

            # a quick evaluation on the whole dataset
            evaluate_all(args, dataloader, ray_fn, model, global_step=global_step, mode='train')
            evaluate_all(args, dataloader, ray_fn, model, global_step=global_step, mode='test')

        global_step += 1

    wandb.alert(
        title='Training Finished',
        text=f'Start to evaluate.',
        level=AlertLevel.WARN)
    args.eval_only = True
    args.denoise = True
    args.grad_normal = True
    evaluate(args)


def eval(args, dataloader, ray_fn, model, img_i=None, mode='test', log_level=2):
    # log_level - 0: return metrics, 1: return metrics and maps, otherwise: wandb logging
    H = dataloader.scene_info['H']
    W = dataloader.scene_info['W']

    if img_i is None:
        i_split = dataloader.i_test if mode.startswith('test') else dataloader.i_train
        img_i = np.random.choice(np.arange(0, len(i_split)))
    inds = img_i * H * W + np.arange(0, H * W)
    rays, targets = dataloader(inds, mode=mode)  # (H*W, C)
    targets['dist_norm'] = torch.zeros_like(targets['dist'])

    # Forward network
    outputs = {}
    with torch.enable_grad():
        for i in range(0, len(rays), args.N_rand):
            batch_inputs, d0, batch_raydirs = get_ray_param(ray_fn, rays[i:i+args.N_rand])
            targets['dist_norm'][i:i+args.N_rand] = (targets['dist'][i:i+args.N_rand] - d0) / (args.radius * 2.)

            outs = model(batch_inputs)
            outs['dist_abs'] = outs['dist'] * (2.*args.radius) + d0
            if 'rgb' in outs:
                outs['rgb_pred'] = torch.sigmoid(outs['rgb'])
            if args.denoise:
                gn = get_surface_gradient(outs['dist'], batch_raydirs)
                outs['not_outlier'] = gn < args.outlier_thres
            if args.grad_normal:
                outs['normal'] = get_surface_normal(outs['dist_abs'], batch_raydirs)

            for k in outs:
                outs[k] = outs[k].detach()  # avoid OOM owing to the gradient
                outputs[k] = outs[k] if k not in outputs else torch.cat([outputs[k], outs[k]], dim=0)

    for k in outputs:
        outputs[k] = outputs[k].reshape([H, W] + list(outputs[k].shape[1:]))
    for k in targets:
        if targets[k] is not None:
            targets[k] = targets[k].reshape([H, W] + list(targets[k].shape[1:]))

    # Metrics
    metrics = {}
    metrics['ade'] = torch.sqrt(((outputs['dist'] - targets['dist_norm'])**2 * targets['mask']).sum() / \
                     targets['mask'].sum()) * (args.radius * 2.)

    if 'rgb' in outputs:
        rgb_pred = outputs['rgb_pred'] * targets['mask'] + (1. - targets['mask'])
        metrics['psnr'] = PSNR(rgb_pred, targets['image'])
        rgb_pred = rgb_pred[None].permute(0,3,1,2)
        rgb_gt = targets['image'][None].permute(0,3,1,2)
        metrics['ssim'] = SSIM(rgb_pred, rgb_gt)
        metrics['lpips'] = LPIPS(2.*rgb_pred-1., 2.*rgb_gt-1.)

    if log_level == 0:
        return metrics

    # Visualization
    vis_results = {}
    mask = targets['mask']
    if args.continuous:
        mask = torch.ones_like(targets['mask'])
    vis_results['dist'] = log.to_distmap(outputs['dist_abs'], mask)
    if args.grad_normal:
        vis_results['normal'] = log.to_normalmap(outputs['normal'], mask)
    if 'rgb_pred' in outputs:
        outputs['rgb_pred'] = outputs['rgb_pred'] * mask + (1. - mask)
        vis_results['color'] = log.to_colormap(outputs['rgb_pred'])

    vis_results['dist_abs'] = outputs['dist_abs'].clone()
    vis_results['dist_abs'][mask==0] = 1e8

    ## remove outliers for tsdf-fusion
    if args.denoise:
        mask = mask * outputs['not_outlier']
        vis_results['dist_abs'][mask==0] = 1e8

    if log_level == 1:
        return metrics, vis_results

    log_dict = {
        f'eval_{mode}/ade': metrics['ade'].item(),
        f'eval_{mode}/distmap': [wandb.Image(vis_results['dist'], caption='pred'),
                                 wandb.Image(log.to_distmap(targets['dist'], targets['mask']), caption='gt')]
    }
    if 'rgb_pred' in outputs:
        log_dict[f'eval_{mode}/psnr'] = metrics['psnr'].item()
        log_dict[f'eval_{mode}/ssim'] = metrics['ssim'].item()
        log_dict[f'eval_{mode}/lpips'] = metrics['lpips'].item()
        log_dict[f'eval_{mode}/colormap'] = [wandb.Image(vis_results['color'], caption='pred'),
                                             wandb.Image(log.to_colormap(targets['image']), caption='gt')]
    if args.grad_normal:
        log_dict[f'eval_{mode}/normalmap'] = [wandb.Image(vis_results['normal'], caption='pred')]

    wandb.log(log_dict)


def evaluate_all(args, dataloader, ray_fn, model, global_step, mode='train'):
    i_split = dataloader.i_train if mode == 'train' else dataloader.i_test
    metrics_all = {}
    for img_i in trange(len(i_split)):
        metrics = eval(args, dataloader, ray_fn, model, img_i=img_i, mode=mode, log_level=0)
        for k in metrics:
            val = metrics[k].item()
            if k not in metrics_all:
                metrics_all[k] = [val]
            else:
                metrics_all[k].append(val)

    log_dict = {}
    for k in metrics_all:
        metrics_all[k] = np.mean(metrics_all[k])
        log_dict[f'eval_{mode}/{k}_all'] = metrics_all[k]
    wandb.log(log_dict)

    log_text = f'[{mode}_{global_step}]'
    for k in metrics_all:
        log_text += f' {k.upper()}={metrics_all[k]:.04f}'
    print(log_text)
    with open(os.path.join(args.logdir, args.expname, f'eval_metrics.txt'), 'a') as file:
        file.write(log_text+'\n')


def evaluate(args):
    # Load dataset and network
    dataloader = Dataloader(args, device)
    ray_fn, global_step, model, model_cls, optimizer, scheduler = create_net(args, dataloader.scene_info, device)

    # Save evaluation results
    modes = ['train', 'test']
    metrics_all = {
        'ADE': {'train': [], 'test': []},
        'CD': {'train': [], 'test': []},
        'CD_median': {'train':[], 'test': []},
        'PSNR': {'train': [], 'test': []},
        'SSIM': {'train': [], 'test': []},
        'LPIPS': {'train': [], 'test': []}
    }

    save_path = os.path.join(args.logdir, args.expname, 'eval')
    for mode in modes:
        os.makedirs(os.path.join(save_path, mode), exist_ok=True)
    f = os.path.join(save_path, f'eval_metrics.txt')

    # read gt mesh
    scene_name = args.datadir.split('/')[-1]
    mesh_gt = trimesh.load_mesh(os.path.join(args.datadir, scene_name + '.ply'))

    for mode in modes:
        volume = pipelines.integration.ScalableTSDFVolume(voxel_length=args.voxel_sz,
                                                sdf_trunc=5.*args.voxel_sz,
                                                color_type=pipelines.integration.TSDFVolumeColorType.RGB8)

        i_split = dataloader.i_train if mode == 'train' else dataloader.i_test
        for img_i in trange(len(i_split)):
            metrics, vis_results = eval(args, dataloader, ray_fn, model, img_i=img_i, mode=mode, log_level=1)

            log_text = f'[{mode} {i_split[img_i]}]'
            for k in metrics:
                val = metrics[k].item()
                metrics_all[k.upper()][mode].append(val)
                log_text += f' {k.upper()}={val:.04f}'
            print(log_text)

            # Save qualitative results
            imageio.imwrite(os.path.join(save_path, mode, f'{i_split[img_i]:06d}_d.png'), vis_results['dist'].squeeze())
            if 'normal' in vis_results:
                imageio.imwrite(os.path.join(save_path, mode, f'{i_split[img_i]:06d}_n.png'), vis_results['normal'])
            if 'color' in vis_results:
                imageio.imwrite(os.path.join(save_path, mode, f'{i_split[img_i]:06d}.png'), vis_results['color'])

            # tsdf integration
            depth = convert_d(vis_results['dist_abs'].squeeze().cpu().numpy(), dataloader.scene_info, out='dep')
            depth = o3d.geometry.Image((depth * 1000.).astype(np.int16))
            color = np.ones((dataloader.scene_info['H'], dataloader.scene_info['W'], 3))
            if 'color' in vis_results:
                color = vis_results['color']
            elif 'normal' in vis_results:
                color = vis_results['normal']
            color = o3d.geometry.Image(color.astype(np.uint8))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale=1000.0, depth_trunc=100., convert_rgb_to_intensity=False)
            cam_pose = dataloader.cam_poses[i_split[img_i]]
            if args.dataset in ['blender', 'dmsr']:
                cam_pose[:3, 1:3] *= -1.
            intrinsic = o3d.camera.PinholeCameraIntrinsic(dataloader.scene_info['W'], dataloader.scene_info['H'],
                                                          dataloader.scene_info['focal'], dataloader.scene_info['focal'],
                                                          dataloader.scene_info['K'][0,2], dataloader.scene_info['K'][1,2])
            volume.integrate(rgbd, intrinsic, np.linalg.inv(cam_pose))

        # export mesh
        m = volume.extract_triangle_mesh()
        m.compute_vertex_normals()
        mesh = trimesh.Trimesh(vertices=np.asarray(m.vertices), faces=np.asarray(m.triangles),
                               vertex_normals=np.asarray(m.vertex_normals), vertex_colors=np.asarray(m.vertex_colors))
        if args.dataset == 'scannet' and args.continuous:
            mesh.export(os.path.join(save_path, f'tsdf_mesh_{mode}_c.obj'))
        else:
            mesh.export(os.path.join(save_path, f'tsdf_mesh_{mode}.obj'))

        # chamfer-distance (m, x1e-3)
        scale = 1e3
        pred_cloud = torch.tensor(mesh.sample(args.cd_sample)[None]).to(device)
        targ_cloud = torch.tensor(mesh_gt.sample(args.cd_sample)[None]).to(device)
        dist1, dist2, _, _ = CD(pred_cloud, targ_cloud)
        cd = (torch.mean(dist1) + torch.mean(dist2)) * scale
        cd_median = (torch.median(dist1) + torch.median(dist2)) * scale
        metrics_all['CD'][mode].append(cd.item())
        metrics_all['CD_median'][mode].append(cd_median.item())

    for mode in modes:
        with open(f, 'a') as file:
            log_text = f'[{mode}]'
            for k in metrics_all:
                v = np.mean(metrics_all[k][mode])
                log_text += f' {k}={v:.04f}'
            print(log_text)
            file.write(log_text+'\n')


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    if args.expname == '':
        args.expname = f'mv{args.N_views}d{args.netdepth}w{args.netwidth}rgb{args.rgb_layer}w{str(args.w_rgb)}' \
                       f'_lr{str(args.lrate)}bs{args.N_rand}iters{int(args.N_iters / 1000)}k'
    args.expname = f'{args.dataset}-{args.scene}_{args.expname}'
    args.datadir = os.path.join(args.datadir, args.dataset, args.scene)
    if args.ckpt_path_cls is None:
        if args.dataset == 'blender':
            args.ckpt_path_cls = f'logs/classifier/{args.dataset}-{args.scene}_d8w512ext1pw0.1_' \
                                 f'lr0.0001bs2048iters50k/0049999.tar'
        elif args.dataset == 'dmsr':
            args.ckpt_path_cls = f'logs/classifier/{args.dataset}-{args.scene}_d8w512ext1pw1.0_' \
                                 f'lr5e-05bs1024iters180k/0179999.tar'
        elif args.dataset == 'scannet':
            args.ckpt_path_cls = f'logs/classifier/{args.dataset}-{args.scene}_d8w512ext1pw1.0_' \
                                 f'lr5e-05bs2048iters130k/0129999.tar'
        else:
            raise NotImplementedError

    if not args.eval_only:
        log.save_config(args)
        train(args)
    else:
        evaluate(args)
