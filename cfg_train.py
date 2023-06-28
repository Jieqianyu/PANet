import datetime
import os
import warnings

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import build_dataloader
from models import build_network
from utils import common_utils, train_utils
from utils.config import global_args, global_cfg, log_config_to_file
from utils.evaluate_panoptic import init_eval, printResults

warnings.filterwarnings("ignore")

def BaseMain(args, cfg):
    if args.launcher == None:
        dist_train = False
    else:
        args.batch_size, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.batch_size, args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True
    cfg['DIST_TRAIN'] = dist_train
    output_dir = os.path.join(args.log_dir, args.tag)
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    tmp_dir = os.path.join(output_dir, 'tmp')
    summary_dir = os.path.join(output_dir, 'summary')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir, exist_ok=True)

    if args.onlyval and args.saveval:
        results_dir = os.path.join(output_dir, 'test', 'sequences')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        for i in range(8, 9):
            sub_dir = os.path.join(results_dir, str(i).zfill(2), 'predictions')
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir, exist_ok=True)

    if args.onlytest:
        results_dir = os.path.join(output_dir, 'test', 'sequences')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        for i in range(11,22):
            sub_dir = os.path.join(results_dir, str(i).zfill(2), 'predictions')
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir, exist_ok=True)

    log_file = os.path.join(output_dir, ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        total_gpus = dist.get_world_size()
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.config, output_dir))

    ### create dataloader
    if (not args.onlytest) and (not args.onlyval):
        train_dataset_loader = build_dataloader(args, cfg, split='train', logger=logger)
        val_dataset_loader = build_dataloader(args, cfg, split='valid', logger=logger, no_shuffle=True)
    elif args.onlyval:
        val_dataset_loader = build_dataloader(args, cfg, split='valid', logger=logger, no_shuffle=True)
    else:
        test_dataset_loader = build_dataloader(args, cfg, split='test', logger=logger, no_shuffle=True)

    ### create model
    model = build_network(cfg)
    model.cuda()

    ### create optimizer
    optimizer = train_utils.build_optimizer(model, cfg)

    ### load ckpt
    ckpt_fname = os.path.join(ckpt_dir, args.ckpt_name)
    epoch = -1

    other_state = {}
    if (args.pretrained_ckpt is not None and os.path.exists(args.pretrained_ckpt)) and os.path.exists(ckpt_fname):
        logger.info("Now in pretrain mode and loading ckpt: {}".format(ckpt_fname))
        optimizer = train_utils.build_optimizer(model, cfg)
        epoch, other_state = train_utils.load_params_with_optimizer_otherstate(model, ckpt_fname, to_cpu=dist_train, optimizer=optimizer, logger=logger) # new feature
        logger.info("Loaded Epoch: {}".format(epoch))
    elif (args.pretrained_ckpt is not None and os.path.exists(args.pretrained_ckpt)):
        logger.info("Loading pretrained model: {}".format(args.pretrained_ckpt))
        train_utils.load_pretrained_model(model, args.pretrained_ckpt, to_cpu=dist_train, logger=logger)
        optimizer = train_utils.build_optimizer(model, cfg)
    elif os.path.exists(ckpt_fname):
        epoch, other_state = train_utils.load_params_with_optimizer_otherstate(model, ckpt_fname, to_cpu=dist_train, optimizer=optimizer, logger=logger) # new feature
        logger.info("Loaded Epoch: {}".format(epoch))
    if other_state is None:
        other_state = {}

    ### create optimizer and scheduler
    lr_scheduler = train_utils.build_scheduler(optimizer, cfg, epoch)
    if lr_scheduler == None:
        logger.info('Not using lr scheduler')

    model.eval()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()], find_unused_parameters=True)
    # logger.info(model)

    if cfg.LOCAL_RANK==0:
        writer = SummaryWriter(log_dir=summary_dir)

    logger.info('**********************Start Training**********************')
    rank = cfg.LOCAL_RANK
    best_before_iou = -1 if 'best_before_iou' not in other_state else other_state['best_before_iou']
    best_pq = -1 if 'best_pq' not in other_state else other_state['best_pq']
    best_after_iou = -1 if 'best_after_iou' not in other_state else other_state['best_after_iou']
    global_iter = 0 if 'global_iter' not in other_state else other_state['global_iter']
    val_global_iter = 0 if 'val_global_iter' not in other_state else other_state ['val_global_iter']
    best_tracking_loss = 10086 if 'best_tracking_loss' not in other_state else other_state ['best_tracking_loss']

    ### test
    if args.onlytest:
        logger.info('----EPOCH {} Testing----'.format(epoch))
        model.eval()
        if rank == 0:
            vbar = tqdm(total=len(test_dataset_loader), dynamic_ncols=True)
        for i_iter, inputs in enumerate(test_dataset_loader):
            with torch.no_grad():
                ret_dict = model(inputs, is_test=True, require_merge=True)
                common_utils.save_test_results(ret_dict, results_dir, inputs)
            if rank == 0:
                vbar.set_postfix({'fname':'/'.join(inputs['pcd_fname'][0].split('/')[-3:])})
                vbar.update(1)
        if rank == 0:
            vbar.close()
        logger.info("----Testing Finished----")
        return

    ### evaluate
    if args.onlyval:
        logger.info('----EPOCH {} Evaluating----'.format(epoch))
        model.eval()
        min_points = 50 # according to SemanticKITTI official rule
        before_merge_evaluator = init_eval(min_points=min_points)
        after_merge_evaluator = init_eval(min_points=min_points)
        if rank == 0:
            vbar = tqdm(total=len(val_dataset_loader), dynamic_ncols=True)
        for i_iter, inputs in enumerate(val_dataset_loader):
            inputs['i_iter'] = i_iter
            # torch.cuda.empty_cache()
            with torch.no_grad():
                ret_dict = model(inputs, is_test=True, before_merge_evaluator=before_merge_evaluator,
                                after_merge_evaluator=after_merge_evaluator, require_merge=True)
                #########################
                # with open('./ipnb/{}_matching_list.pkl'.format(i_iter), 'wb') as fd:
                #     pickle.dump(ret_dict['matching_list'], fd)
                #########################
                if args.saveval:
                    common_utils.save_test_results(ret_dict, results_dir, inputs)
            if rank == 0:
                vbar.set_postfix({'loss': ret_dict['loss'].item(),
                                  'fname':'/'.join(inputs['pcd_fname'][0].split('/')[-3:]),
                                  'ins_num': -1 if 'ins_num' not in ret_dict else ret_dict['ins_num']})
                vbar.update(1)
        if dist_train:
            before_merge_evaluator = common_utils.merge_evaluator(before_merge_evaluator, tmp_dir)
            dist.barrier()
            after_merge_evaluator = common_utils.merge_evaluator(after_merge_evaluator, tmp_dir)

        if rank == 0:
            vbar.close()
        if rank == 0:
            ## print results
            logger.info("Before Merge Semantic Scores")
            before_merge_results = printResults(before_merge_evaluator, logger=logger, sem_only=True)
            logger.info("After Merge Panoptic Scores")
            after_merge_results = printResults(after_merge_evaluator, logger=logger)

        logger.info("----Evaluating Finished----")
        return

    ### train
    while True:
        epoch += 1
        if 'MAX_EPOCH' in cfg.OPTIMIZE.keys():
            if epoch > cfg.OPTIMIZE.MAX_EPOCH:
                break

        ### train one epoch
        logger.info('----EPOCH {} Training----'.format(epoch))
        loss_acc = 0
        if rank == 0:
            pbar = tqdm(total=len(train_dataset_loader), dynamic_ncols=True)
        for i_iter, inputs in enumerate(train_dataset_loader):
            # torch.cuda.empty_cache()
            torch.autograd.set_detect_anomaly(True)
            model.eval()
            optimizer.zero_grad()
            inputs['i_iter'] = i_iter
            inputs['rank'] = rank
            inputs['epoch'] = epoch - 1
            ret_dict = model(inputs)
            loss = ret_dict['loss']
            loss.backward()
            optimizer.step()

            if rank == 0:
                try:
                    cur_lr = float(optimizer.lr)
                except:
                    cur_lr = optimizer.param_groups[0]['lr']
                loss_acc += loss.item()
                pbar.set_postfix({'loss': loss.item(), 'lr': cur_lr, 'mean_loss': loss_acc / float(i_iter+1)})
                pbar.update(1)

                writer.add_scalar('Train/LR', cur_lr, global_iter)
                more_keys = []
                for k, _ in ret_dict.items():
                    if k.find('loss') != -1:
                        more_keys.append(k)
                for ki, k in enumerate(more_keys):
                    ki += 1
                    writer.add_scalar('Train/{}_{}'.format(str(ki).zfill(2), k), ret_dict[k], global_iter)
                global_iter += 1
        if rank == 0:
            pbar.close()

        ### evaluate after each epoch
        logger.info('----EPOCH {} Evaluating----'.format(epoch))
        model.eval()
        min_points = 50
        before_merge_evaluator = init_eval(min_points=min_points)
        after_merge_evaluator = init_eval(min_points=min_points)
        tracking_loss = 0
        if rank == 0:
            vbar = tqdm(total=len(val_dataset_loader), dynamic_ncols=True)
        for i_iter, inputs in enumerate(val_dataset_loader):
            # torch.cuda.empty_cache()
            inputs['i_iter'] = i_iter
            inputs['rank'] = rank
            inputs['epoch'] = epoch - 1
            with torch.no_grad():
                ret_dict = model(inputs, is_test=True, before_merge_evaluator=before_merge_evaluator,
                                     after_merge_evaluator=after_merge_evaluator, require_merge=True)
            if rank == 0:
                vbar.set_postfix({'loss': ret_dict['loss'].item()})
                vbar.update(1)
                more_keys = []
                for k, _ in ret_dict.items():
                    if k.find('loss') != -1:
                        more_keys.append(k)
                for ki, k in enumerate(more_keys):
                    ki += 1
                    writer.add_scalar('Val/{}_{}'.format(str(ki).zfill(2), k), ret_dict[k], val_global_iter)
                val_global_iter += 1
        tracking_loss /= len(val_dataset_loader)
        if dist_train:
            try:
                before_merge_evaluator = common_utils.merge_evaluator(before_merge_evaluator, tmp_dir, prefix='before_')
                dist.barrier()
                after_merge_evaluator = common_utils.merge_evaluator(after_merge_evaluator, tmp_dir, prefix='after_')
            except:
                print("Someting went wrong when merging evaluator in rank {}".format(rank))
        if rank == 0:
            vbar.close()
        if rank == 0:
            ## print results
            logger.info("Before Merge Semantic Scores")
            before_merge_results = printResults(before_merge_evaluator, logger=logger, sem_only=True)
            logger.info("After Merge Panoptic Scores")
            after_merge_results = printResults(after_merge_evaluator, logger=logger)
            ## save ckpt
            other_state = {
                'best_before_iou': best_before_iou,
                'best_pq': best_pq,
                'best_after_iou': best_after_iou,
                'global_iter': global_iter,
                'val_global_iter': val_global_iter,
                'best_tracking_loss': best_tracking_loss,
            }
            saved_flag = False
            if best_tracking_loss > tracking_loss:
                best_tracking_loss = tracking_loss
                if not saved_flag:
                    states = train_utils.checkpoint_state(model, optimizer, epoch, other_state)
                    train_utils.save_checkpoint(states, os.path.join(ckpt_dir,
                        'checkpoint_epoch_{}_{}.pth'.format(epoch, str(tracking_loss)[:5])))
                    saved_flag = True
            if best_before_iou < before_merge_results['iou_mean']:
                best_before_iou = before_merge_results['iou_mean']
                if not saved_flag:
                    states = train_utils.checkpoint_state(model, optimizer, epoch, other_state)
                    train_utils.save_checkpoint(states, os.path.join(ckpt_dir,
                        'checkpoint_epoch_{}_{}_{}_{}.pth'.format(epoch, str(best_before_iou)[:5], str(best_pq)[:5], str(best_after_iou)[:5])))
                    saved_flag = True
            if best_pq < after_merge_results['pq_mean']:
                best_pq = after_merge_results['pq_mean']
                if not saved_flag:
                    states = train_utils.checkpoint_state(model, optimizer, epoch, other_state)
                    train_utils.save_checkpoint(states, os.path.join(ckpt_dir,
                        'checkpoint_epoch_{}_{}_{}_{}.pth'.format(epoch, str(best_before_iou)[:5], str(best_pq)[:5], str(best_after_iou)[:5])))
                    saved_flag = True
            if best_after_iou < after_merge_results['iou_mean']:
                best_after_iou = after_merge_results['iou_mean']
                if not saved_flag:
                    states = train_utils.checkpoint_state(model, optimizer, epoch, other_state)
                    train_utils.save_checkpoint(states, os.path.join(ckpt_dir,
                        'checkpoint_epoch_{}_{}_{}_{}.pth'.format(epoch, str(best_before_iou)[:5], str(best_pq)[:5], str(best_after_iou)[:5])))
                    saved_flag = True
            logger.info("Current best before IoU: {}".format(best_before_iou))
            logger.info("Current best after IoU: {}".format(best_after_iou))
            logger.info("Current best after PQ: {}".format(best_pq))
            logger.info("Current best tracking loss: {}".format(best_tracking_loss))
        if lr_scheduler != None:
            lr_scheduler.step(epoch) # new feature

if __name__ =='__main__':
    args, cfg = global_args, global_cfg
    BaseMain(args, cfg)