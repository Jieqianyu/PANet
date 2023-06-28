import torch
from torch.utils.data import DistributedSampler as _DistributedSampler

from utils import common_utils

from .semantic_kitti import SemanticKitti

__all_dataset__ =  {
    'SemanticKitti': SemanticKitti,
}

def collate_fn(data):
    keys = data[0].keys()
    out_dict = {}
    for key in keys:
        out_dict[key] = [d[key] for d in data]
    return out_dict

class DistributedSampler(_DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

def build_dataloader(args, cfg, split='train', logger=None, no_shuffle=False):
    if logger is not None:
        logger.info("Building dataloader for {} set.".format(split))

    is_training = (split == 'train')
    if cfg.DATA_CONFIG.DATASET_NAME == 'SemanticKitti':
        dataset=__all_dataset__[cfg.DATA_CONFIG.DATASET_NAME](
            data_root=cfg.DATA_CONFIG.DATASET_PATH,
            data_config_file=cfg.DATA_CONFIG.DATASET_CONFIG,
            setname=split,
            lims=cfg.MODEL.LIMS,
            augmentation= True if is_training else False,
            with_gt = True if split != 'test' else False
        )
    else:
        raise NotImplementedError

    if cfg.DIST_TRAIN:
        if is_training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=is_training and (not no_shuffle))
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank)
    else:
        sampler = None
    dataset_loader = torch.utils.data.DataLoader(
        dataset = dataset,
        collate_fn=collate_fn,
        batch_size = args.batch_size,
        shuffle = ((sampler is None) and is_training) and (not no_shuffle),
        num_workers = cfg.DATA_CONFIG.DATALOADER.NUM_WORKER,
        pin_memory = True,
        drop_last = True if is_training else False,
        sampler = sampler,
        timeout = 0
    )
    if logger is not None:
        logger.info("Shuffle: {}".format(is_training and (not no_shuffle)))

    return dataset_loader