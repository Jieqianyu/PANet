import os
import random

import numpy as np
import torch
import yaml


def mask_op(data, x_min, x_max):
    mask = (data > x_min) & (data < x_max)
    return mask


def get_mask(pc, lims):
    mask_x = mask_op(pc[:, 0], lims[0][0] + 0.0001, lims[0][1] - 0.0001)
    mask_y = mask_op(pc[:, 1], lims[1][0] + 0.0001, lims[1][1] - 0.0001)
    mask_z = mask_op(pc[:, 2], lims[2][0] + 0.0001, lims[2][1] - 0.0001)
    mask = (mask_x) & (mask_y) & mask_z
    return mask

def get_polar_mask(pc, lims):
    r = np.sqrt(pc[:,0]**2 + pc[:,1]**2)
    mask_x = mask_op(r, lims[0][0] + 0.0001, lims[0][1] - 0.0001)
    mask_z = mask_op(pc[:, 2], lims[2][0] + 0.0001, lims[2][1] - 0.0001)
    mask = (mask_x) & mask_z
    return mask


EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']


def is_scan(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_label(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


def rotate_pc(pc, theta):
    rotation_matrix = np.array([np.cos([theta]),
                                np.sin([theta]),
                                -np.sin([theta]),
                                np.cos([theta])], dtype=np.float).reshape(2, 2)

    rxy = np.matmul(pc, rotation_matrix)

    return rxy


def random_rotate_pc(pc, xlim, ylim, theta):
    center_x = (xlim[0] + xlim[1]) / 2.
    center_y = (ylim[0] + ylim[1]) / 2.
    center = np.array([center_x, center_y], dtype=np.float)
    x = pc[:, 0] - center_x
    y = pc[:, 1] - center_y

    xy = np.stack([x, y], axis=-1)
    rxy = rotate_pc(xy, theta)

    add = rxy + center

    return add


def augmentation_rotate_pc(pc, lims):
    # angle = np.random.uniform() * np.pi * 2 #- np.pi
    angle = (-np.pi - np.pi) * torch.rand(1).tolist()[0] + np.pi
    rxy = random_rotate_pc(pc, lims[0], lims[1], angle)
    # print(pc[:, 0].size(), rxy[0].size())
    pc[:, :2] = rxy

    return pc


def augmentation_random_flip(pc):
    flip_type = torch.randint(4, (1,)).tolist()[0]
    if flip_type==1:
        pc[:,0] = -pc[:,0]
    elif flip_type==2:
        pc[:,1] = -pc[:,1]
    elif flip_type==3:
        pc[:,:2] = -pc[:,:2]
    return pc

def augmentation_scale(pc):
    noise_scale = np.random.uniform(0.95, 1.05)
    noise_scale = (0.95 - 1.05) * torch.rand(1).tolist()[0] + 1.05
    pc[:, 0] = noise_scale * pc[:, 0]
    pc[:, 1] = noise_scale * pc[:, 1]
    pc[:, 2] = noise_scale * pc[:, 2]
    # drop = prob
    return pc


def augmentation_rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1,0,0],
                    [0,np.cos(angles[0]),-np.sin(angles[0])],
                    [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                    [0,1,0],
                    [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                    [np.sin(angles[2]),np.cos(angles[2]),0],
                    [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))
    out = np.dot(batch_data[:, :3].reshape((-1, 3)), R)
    batch_data[:, :3] = out
    return batch_data

def random_jitter_point_cloud(pc, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, jittered batch of point clouds
    """
    N, C = pc.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, 3), -1*clip, clip)
    # jittered_data += batch_data
    pc[:, :3] += jittered_data
    return pc

things_sem = set([1, 2, 3, 4, 5, 6, 7, 8])
def aggregate_pointwise_center_offset(offsets, xyz, sem_labels, ins_labels, center_type, min_points=10):
    # ins_num = np.max(ins_labels) + 1
    # for i in range(1, ins_num):
    centers, ids, classes = [], [], []
    for i in np.unique(ins_labels):
        i_indices = (ins_labels == i).reshape(-1)
        xyz_i = xyz[i_indices]
        sem = np.unique(sem_labels[i_indices])[0]
        if sem not in things_sem:
            continue
        if xyz_i.shape[0] <= min_points:
            continue
        if center_type == 'Axis_center':
            raise NotImplementedError
        elif center_type == 'Mass_center':
            mean_xyz = np.mean(xyz_i, axis=0)
        else:
            raise NotImplementedError
        offsets[i_indices] = mean_xyz - xyz_i
        centers.append(mean_xyz)
        ids.append(np.unique(ins_labels[i_indices])[0])
        classes.append(sem)

    return offsets, centers, ids, classes


class SemanticKitti(torch.utils.data.Dataset):
    CLASSES = ('unlabeled',
               'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
               'person', 'bicyclist', 'motorcyclist', 'road',
               'parking', 'sidewalk', 'other-ground', 'building', 'fence',
               'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign')

    def __init__(self, data_root, data_config_file, setname,
                 lims,
                 augmentation=False,
                 max_num=140000,
                 with_gt=False,
                 ignore_class=[0],
                 shuffle_index=False,
                 prefetch=False):
        print(setname, augmentation)
        self.prefetch = prefetch
        self.data_root = data_root
        self.data_config = yaml.safe_load(open(data_config_file, 'r'))
        self.sequences = self.data_config["split"][setname]
        self.setname = setname
        self.labels = self.data_config['labels']
        self.learning_map = self.data_config["learning_map"]
        print(self.learning_map)
        self.learning_map_inv = self.data_config["learning_map_inv"]
        self.with_gt = with_gt
        self.color_map = self.data_config['color_map']

        self.lims = lims
        self.augmentation = augmentation
        self.scan_files = []
        self.label_files = []
        self.shuffle_index = shuffle_index
        self.ignore_class = ignore_class
        # fill in with names, checking that all sequences are complete
        for seq in self.sequences:
            # to string
            seq = '{0:02d}'.format(int(seq))

            print("parsing seq {}".format(seq))

            # get paths for each
            scan_path = os.path.join(self.data_root, seq, "velodyne")
            label_path = os.path.join(self.data_root, seq, "labels")

            # get files
            scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
            label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(label_path)) for f in fn if is_label(f)]

            # check all scans have labels
            if self.with_gt:
                assert (len(scan_files) == len(label_files))

            # extend list
            self.scan_files.extend(scan_files)
            self.label_files.extend(label_files)

        self.scan_files.sort()
        self.label_files.sort()
        self.file_idx = np.arange(0, len(self.scan_files))
        self.num_files_ = len(self.file_idx)
        print("Using {} scans from sequences {}".format(len(self.scan_files),
                                                        self.sequences))
        print(self.augmentation, " is aug")

        # get class distribution weight
        epsilon_w = 0.001
        origin_class = self.data_config['content'].keys()
        weights = np.zeros((len(self.data_config['learning_map_inv'])-1,),dtype = np.float32)
        for class_num in origin_class:
            if self.data_config['learning_map'][class_num] != 0:
                weights[self.data_config['learning_map'][class_num]-1] += self.data_config['content'][class_num]
        self.CLS_LOSS_WEIGHT = 1/(weights + epsilon_w)

        self._set_group_flag()


    def __len__(self):
        return self.num_files_

    def get_n_classes(self):
        return len(self.learning_map_inv)

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    @staticmethod
    def map(label, mapdict):
        # put label from original values to xentropy
        # or vice-versa, depending on dictionary values
        # make learning map a lookup table
        maxkey = 0
        for key, data in mapdict.items():
            if isinstance(data, list):
                nel = len(data)
            else:
                nel = 1
            if key > maxkey:
                maxkey = key
        # +100 hack making lut bigger just in case there are unknown labels
        if nel > 1:
            lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
        else:
            lut = np.zeros((maxkey + 100), dtype=np.int32)
        for key, data in mapdict.items():
            try:
                lut[key] = data
            except IndexError:
                print("Wrong key ", key)
        # do the mapping
        return lut[label]

    def to_color(self, label):
        # put label in original values
        label = SemanticKitti.map(label, self.learning_map_inv)
        # put label in color
        return SemanticKitti.map(label, self.color_map)

    def get_xentropy_class_string(self, idx):
        return self.labels[self.learning_map_inv[idx]]

    def __getitem__(self, idx):
        return self.get_normal_item(idx)

    def get_normal_item(self, idx):
        if idx >= self.num_files_:
            np.random.shuffle(self.file_idx)
        scan_file = self.scan_files[self.file_idx[idx]]
        scan = np.fromfile(scan_file, dtype=np.float32)
        scan = scan.reshape((-1, 4))

        if self.with_gt:
            label_file = self.label_files[self.file_idx[idx]]
            label = np.fromfile(label_file, dtype=np.uint32)
            label = label.reshape((-1))
            sem_label = label & 0xFFFF  # semantic label in lower half
            inst_label = label
            sem_label = self.map(sem_label, self.learning_map)
            inst_label = inst_label.astype(np.int32)

        if self.shuffle_index:
            pt_idx = np.random.permutation(np.arange(0, scan.shape[0]))
            scan = scan[pt_idx]
            if self.with_gt:
                sem_label = sem_label[pt_idx]
                inst_label = inst_label[pt_idx]
        if self.augmentation:
            scan = augmentation_random_flip(scan)
            scan = augmentation_rotate_pc(scan, self.lims)
            scan = augmentation_scale(scan)
            scan = random_jitter_point_cloud(scan)
            scan = augmentation_rotate_perturbation_point_cloud(scan)
        if self.lims:
            filter_mask = get_mask(scan, self.lims)
            ori_num = filter_mask.shape[0]
            filter_scan = scan[filter_mask]
            if self.with_gt:
                filter_label = sem_label[filter_mask]
                inst_filter_label = inst_label[filter_mask]
        else:
            filter_scan = scan
            if self.with_gt:
                filter_label = sem_label
                inst_filter_label = inst_label

            filter_mask = np.ones(filter_scan.shape[0], np.bool)
        if self.with_gt:
            offsets = np.zeros([filter_scan.shape[0], 3], dtype=np.float32)
            offsets, centers, ins_ids, ins_classes = aggregate_pointwise_center_offset(offsets, filter_scan[:, :3], filter_label, inst_filter_label, 'Mass_center')
            if len(centers) == 0:
                centers, ins_ids, ins_classes = None, None, None
            else:
                centers = np.stack(centers)
                ins_ids = np.stack(ins_ids)
                ins_classes = np.stack(ins_classes)

        if self.with_gt:
            filter_label = torch.from_numpy(filter_label).int()
            inst_filter_label = torch.from_numpy(inst_filter_label).int()
        scan_th = torch.as_tensor(filter_scan, dtype=torch.float32)

        if self.with_gt:
            offsets = torch.as_tensor(offsets, dtype=torch.float32)
            centers = torch.as_tensor(centers, dtype=torch.float32) if centers is not None else None
            ins_ids = torch.from_numpy(ins_ids).int() if ins_ids is not None else None
            ins_classes = torch.from_numpy(ins_classes).int() if ins_classes is not None else None
            if ins_classes is None:
                idx = random.choice(range(self.__len__()))
                return self.get_normal_item(idx)
        filter_mask = torch.from_numpy(filter_mask).bool()

        if self.with_gt:
            input_dict = dict(points=scan_th, filter_mask=filter_mask, points_offset = offsets,
                              points_label=filter_label, inst_label=inst_filter_label, inst_centers=centers, inst_ids=ins_ids, inst_classes=ins_classes,
                              pcd_fname=scan_file)
        else:
            input_dict = dict(points=scan_th, filter_mask=filter_mask, pcd_fname=scan_file)
        example = input_dict
        return example
    def evaluate(self, results, logger=None):
        return NotImplemented