import os
import torch
from collections import defaultdict
from optim.misc import build_optimizer
from models.MA2Trans_net import MA2TransNet
from data.gtlabelpcd_dataset import GTLabelPcdDataset, gtlabelpcd_collate_fn
from torch.utils.data import DataLoader
from optim import get_lr_sched_decay_rate
from utils.logger import LOGGER, TB_LOGGER, AverageMeter, add_log_to_file
from parser import load_parser
from utils.save import ModelSaver

def main(args):
    ### cuda implementation ###
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### data loading  ###
    train_dataset = GTLabelPcdDataset(
        args.train_scan_split, args.anno_file,
        args.scan_dir, args.category_file,
        cat2vec_file=args.cat2vec_file,
        random_rotate=False,
        max_txt_len=args.max_txt_len, max_obj_len=args.max_obj_len,
        keep_background=False,
        num_points=args.num_points, in_memory=True,
        gt_scan_dir=None,
        iou_replace_gt=0,
    )
    val_dataset = GTLabelPcdDataset(
        args.val_scan_split, args.anno_file,
        args.scan_dir, args.category_file,
        cat2vec_file=args.cat2vec_file,
        max_txt_len=None, max_obj_len=None, random_rotate=False,
        keep_background=False,
        num_points=args.num_points, in_memory=True,
        gt_scan_dir=None,
        iou_replace_gt=0,
    )
    collate_fn = gtlabelpcd_collate_fn
    LOGGER.info('train #scans %d, #data %d' % (len(train_dataset.scan_ids), len(train_dataset)))
    LOGGER.info('val #scans %d, #data %d' % (len(val_dataset.scan_ids), len(val_dataset)))

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=False, prefetch_factor=1,)
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=True, drop_last=False, prefetch_factor=1,)

    args.num_train_steps = len(train_dataloader) * args.num_epoch
    LOGGER.info("  Batch size = %d", args.batch_size)
    LOGGER.info("  Num epoch = %d, num steps = %d", args.num_epoch, args.num_train_steps)

    ### Prepare model and optimizer###
    model = MA2TransNet(args)
    model = model.to(device)
    # print(model)
    optimizer, init_lrs = build_optimizer(model, args)

    ### model training ###
    # to compute training statistics
    avg_metrics = defaultdict(AverageMeter)
    global_step = 0

    ### model saving ###
    val_best_scores = {'epoch': -1, 'acc/og3d': -float('inf')}
    model_saver = ModelSaver(os.path.join(args.output_dir, 'ckpts'))
    add_log_to_file(os.path.join(args.output_dir, 'logs', 'log.txt'))

    model.train()
    optimizer.zero_grad()
    optimizer.step()
    for epoch in range(args.num_epoch):
        for ib, batch in enumerate(train_dataloader):
            batch_size = len(batch['scan_ids'])
            result, losses = model(batch)
            losses['total'].backward()

            # optimizer update and logging
            global_step += 1
            # learning rate scheduling:
            lr_decay_rate = get_lr_sched_decay_rate(global_step, args)
            for kp, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = lr_this_step = init_lrs[kp] * lr_decay_rate
            TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

            # log loss
            loss_dict = {'loss/%s' % lk: lv.data.item() for lk, lv in losses.items()}
            for lk, lv in loss_dict.items():
                avg_metrics[lk].update(lv, n=batch_size)
            TB_LOGGER.log_scalar_dict(loss_dict)
            TB_LOGGER.step()

            # update model params
            if args.grad_norm != -1:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_norm)
                TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
            optimizer.step()
            optimizer.zero_grad()
        LOGGER.info(
            'Epoch %d, lr: %.6f, %s', epoch + 1,
            optimizer.param_groups[-1]['lr'],
            ', '.join(['%s: %.4f' % (lk, lv.avg) for lk, lv in avg_metrics.items()])
        )
        if (epoch+1) % 5 == 0:
            LOGGER.info(f'------Epoch {epoch + 1}: start validation (val)------')
            val_log = validate(model, args, val_dataloader)
            TB_LOGGER.log_scalar_dict(
                {f'valid/{k}': v.avg for k, v in val_log.items()}
            )
            output_model_file = model_saver.save(
                model, epoch + 1, optimizer=optimizer, save_latest_optim=True
            )
            if val_log['acc/og3d'].avg > val_best_scores['acc/og3d']:
                val_best_scores['acc/og3d'] = val_log['acc/og3d'].avg
                val_best_scores['epoch'] = epoch + 1
                model_saver.remove_previous_models(epoch+1)
            else:
                os.remove(output_model_file)
    LOGGER.info('Finished training!')
    LOGGER.info(
        'best epoch: %d, best acc/og3d %.4f', val_best_scores['epoch'], val_best_scores['acc/og3d']
    )

@torch.no_grad()
def validate(model, args, val_dataloader):
    model.eval()
    avg_metrics = defaultdict(AverageMeter)
    for ib, batch in enumerate(val_dataloader):
        batch_size = len(batch['scan_ids'])
        result, losses = model(batch, compute_loss=True, is_test=True)

        loss_dict = {'loss/%s' % lk: lv.data.item() for lk, lv in losses.items()}
        for lk, lv in loss_dict.items():
            avg_metrics[lk].update(lv, n=batch_size)

        og3d_preds = torch.argmax(result['og3d_logits'], dim=1).cpu()
        avg_metrics['acc/og3d'].update(
            torch.mean((og3d_preds == batch['tgt_obj_idxs']).float()).item(),
            n=batch_size
        )
        avg_metrics['acc/og3d_class'].update(
            torch.mean((batch['obj_classes'].gather(1, og3d_preds.unsqueeze(1)).squeeze(1) == batch[
                'tgt_obj_classes']).float()).item(),
            n=batch_size
        )
        if args.obj3d_clf_loss:
            obj3d_clf_preds = torch.argmax(result['obj3d_clf_logits'], dim=2).cpu()
            avg_metrics['acc/obj3d_clf'].update(
                (obj3d_clf_preds[batch['obj_masks']] == batch['obj_classes'][batch['obj_masks']]).float().mean().item(),
                n=batch['obj_masks'].sum().item()
            )
        if args.obj3d_clf_pre_loss:
            obj3d_clf_preds = torch.argmax(result['obj3d_clf_pre_logits'], dim=2).cpu()
            avg_metrics['acc/obj3d_clf_pre'].update(
                (obj3d_clf_preds[batch['obj_masks']] == batch['obj_classes'][batch['obj_masks']]).float().mean().item(),
                n=batch['obj_masks'].sum().item()
            )
        if args.txt_clf_loss:
            txt_clf_preds = torch.argmax(result['txt_clf_logits'], dim=1).cpu()
            avg_metrics['acc/txt_clf'].update(
                (txt_clf_preds == batch['tgt_obj_classes']).float().mean().item(),
                n=batch_size
            )
    LOGGER.info(', '.join(['%s: %.4f'%(lk, lv.avg) for lk, lv in avg_metrics.items()]))
    model.train()
    return avg_metrics

if __name__ == '__main__':
    args = load_parser()
    main(args)

