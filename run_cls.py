import os
from tqdm import trange
import numpy as np
from config import config_parser
import torch
import torch.nn.functional as F
import wandb
from wandb import AlertLevel

from utils.dataloader import Dataloader
from utils import log
from utils.ray import get_ray_param
from net_classifier.network import create_classifier
from net_classifier.sampler import get_reference_rays

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
np.random.seed(0)

from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
binary_acc = BinaryAccuracy().to(device)
binary_f1 = BinaryF1Score().to(device)


def train(args):
    # Load dataset
    dataloader = Dataloader(args, device)

    # Create rayparam function and classifer
    ray_fn, global_step, model, optimizer, scheduler = create_classifier(args, dataloader.scene_info, device)
    global binary_acc
    global binary_f1
    binary_acc = BinaryAccuracy(args.vis_thres).to(device)
    binary_f1 = BinaryF1Score(args.vis_thres).to(device)

    # Create experiment logger
    wandb.init(project="RayDF-Classifier")
    wandb.run.name = args.expname
    wandb.watch(model, log="all")

    start = global_step
    train_num = len(dataloader.dists['train_fg'])
    inds = np.random.permutation(train_num)
    step_batch = train_num // args.N_rand

    for i in trange(start, args.N_iters):
        optimizer.zero_grad()
        j = (i-start) % step_batch
        ep = (i-start) // step_batch
        # re-random train indices at the start of each epoch
        if j == 0 and i != start:
            inds = np.random.permutation(train_num)

        # =================== Query Rays ========================
        # Random rays from all foreground rays
        train_i = inds[j * args.N_rand: (j + 1) * args.N_rand]

        # load query rays
        batch_rays, target_dict = dataloader(inds=train_i, mode='train_fg')
        batch_inputs, _, _ = get_ray_param(ray_fn, batch_rays)

        # normalize query surface point
        batch_pts = batch_rays[..., :3] + target_dict['dist'] * batch_rays[..., 3:]
        for c in range(batch_pts.shape[-1]):
            batch_pts[..., c] -= dataloader.scene_info['sphere_center'][c]
        target_dict['pts_norm'] = batch_pts / dataloader.scene_info['sphere_radius']

        # ================= Reference Rays for Visibility Classifier =====================
        ref_rays, cls_targets = get_reference_rays(args, batch_rays, target_dict['dist'],
                                                   dataloader.all_dists[dataloader.i_train],
                                                   dataloader.cam_poses[dataloader.i_train],
                                                   dataloader.scene_info)
        ref_inputs, _, _ = get_ray_param(ray_fn, ref_rays)
        cls_inputs = [batch_inputs[:, None].expand_as(ref_inputs), ref_inputs,
                      target_dict['pts_norm'][:, None].expand_as(ref_inputs[..., :3])]
        cls_outputs = {'vis': model(cls_inputs)}
        cls_outputs['vis_score'] = torch.sigmoid(cls_outputs['vis'])

        # ================= Optimization =====================
        pos_weight = args.pos_weight if args.pos_weight > 0 else (cls_targets==0).sum()/(cls_targets==1).sum()
        pos_weight = torch.tensor([pos_weight]).to(device)
        loss = F.binary_cross_entropy_with_logits(cls_outputs['vis'], cls_targets, pos_weight=pos_weight)
        loss.backward()

        if args.grad_clip > 0.:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()
        new_lrate = optimizer.param_groups[0]['lr']
        scheduler.step()


        # ================= Logging ==========================
        if i % args.i_print == 0 and i != 0:
            acc = binary_acc(cls_outputs['vis_score'], cls_targets)
            f1 = binary_f1(cls_outputs['vis_score'], cls_targets)
            wandb.log({
                'train/_ep': ep,
                'train/_lr': new_lrate,
                'train/loss': loss.item(),
                'train/acc': acc.item(),
                'train/f1': f1.item()
            })

        # ================= Evaluation =====================
        if i % args.i_img == 0 and i != 0:
            eval(args, batch_rays, batch_inputs, target_dict,
                 dataloader, ray_fn, model, dataloader.i_train, mode='train_fg')
            eval(args, batch_rays, batch_inputs, target_dict,
                 dataloader, ray_fn, model, dataloader.i_test, mode='test_fg')

        # Save checkpoints
        if (i != start and i % args.i_weights == 0 and i != 0) or (i + 1) == args.N_iters:
            path = os.path.join(args.logdir, args.expname, '{:07d}.tar'.format(i))
            ckpt_dict = {
                'global_step': global_step,
                'network_fn': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(ckpt_dict, path)
            print('Saved checkpoints at', path)

        global_step += 1

    wandb.alert(
        title='Training Finished',
        text=f'Start to evaluate.',
        level=AlertLevel.WARN)
    args.eval_only = True
    evaluate(args)


def eval(args, batch_rays, batch_inputs, target_dict, dataloader, ray_fn, model, i_split, mode='test_fg'):
    model.eval()
    with torch.no_grad():
        ref_rays, cls_targets = get_reference_rays(args, batch_rays, target_dict['dist'],
                                                   dataloader.all_dists[i_split], dataloader.cam_poses[i_split],
                                                   dataloader.scene_info)
        ref_inputs, _, _ = get_ray_param(ray_fn, ref_rays)
        cls_inputs = [batch_inputs[:, None].expand_as(ref_inputs),
                      ref_inputs,
                      target_dict['pts_norm'][:, None].expand_as(ref_inputs[..., :3])]

        cls_outputs = {'vis': model(cls_inputs)}
        cls_outputs['vis_score'] = torch.sigmoid(cls_outputs['vis'])

        acc = binary_acc(cls_outputs['vis_score'], cls_targets)
        f1 = binary_f1(cls_outputs['vis_score'], cls_targets)
        if not args.eval_only:
            wandb.log({
                f'eval_{mode}/acc': acc.detach().item(),
                f'eval_{mode}/f1': f1.detach().item()
            })
        else:
            return acc, f1
    model.train()
    torch.cuda.empty_cache()


def evaluate(args):
    # Load dataset and network
    dataloader = Dataloader(args, device)
    ray_fn, _, model, _, _ = create_classifier(args, dataloader.scene_info, device)

    # Save evaluation results
    ''' Different modes:
    - train:   query rays: train set | ref rays: train set
    - test:    query rays: train set | ref rays: test set
    - test_v2: query rays: test set  | ref rays: test set
    '''
    modes = ['train', 'test', 'test_v2']
    metrics = {
        'ACC': {'train': [], 'test': [], 'test_v2': []},
        'F1': {'train': [], 'test': [], 'test_v2': []}
    }

    save_path = os.path.join(args.logdir, args.expname, 'eval')
    os.makedirs(save_path, exist_ok=True)
    f = os.path.join(save_path, f'eval_metrics.txt')

    model.eval()
    for mode in modes:
        i_split = dataloader.i_train if mode.startswith('train') else dataloader.i_test
        split = ('test' if mode == 'test_v2' else 'train') + '_fg'
        train_num = len(dataloader.dists[split])
        inds = np.arange(train_num)

        for j in trange(0, train_num, args.N_rand):
            train_i = inds[j:j+args.N_rand]
            batch_rays, target_dict = dataloader(inds=train_i, mode=split)
            batch_inputs, _, _ = get_ray_param(ray_fn, batch_rays)

            # normalize query surface point
            batch_pts = batch_rays[..., :3] + target_dict['dist'] * batch_rays[..., 3:]
            for c in range(batch_pts.shape[-1]):
                batch_pts[..., c] -= dataloader.scene_info['sphere_center'][c]
            target_dict['pts_norm'] = batch_pts / dataloader.scene_info['sphere_radius']

            acc, f1 = eval(args, batch_rays, batch_inputs, target_dict,
                           dataloader, ray_fn, model, i_split, mode=split)
            print(mode, j, 'acc=', acc.item(), 'f1=', f1.item())
            metrics['ACC'][mode].append(acc.item())
            metrics['F1'][mode].append(f1.item())

    for mode in modes:
        with open(f, 'a') as file:
            acc_mean = np.mean(metrics['ACC'][mode])
            f1_mean = np.mean(metrics['F1'][mode])
            print(f'[{mode}] ACC={acc_mean:.04f}, F1={f1_mean:.04f}')
            file.write(f'[{mode}] ACC={acc_mean:.04f}, F1={f1_mean:.04f}\n')


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    if args.expname == '':
        args.expname = f'd{args.netdepth_cls}w{args.netwidth_cls}ext{args.ext_layer_cls}pw{str(args.pos_weight)}' \
                       f'_lr{str(args.lrate)}bs{args.N_rand}iters{int(args.N_iters/1000)}k'
    args.expname = f'{args.dataset}-{args.scene}_{args.expname}'
    args.datadir = os.path.join(args.datadir, args.dataset, args.scene)

    if not args.eval_only:
        log.save_config(args)
        train(args)
    else:
        evaluate(args)

