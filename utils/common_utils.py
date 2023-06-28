import numpy as np
import torch
import random
import logging
import os
import torch.multiprocessing as mp
import torch.distributed as dist
import subprocess
import pickle
import shutil
from scipy import stats as s
import numba as nb
from .evaluate_panoptic import class_inv_lut, is_nuscenes

def SemKITTI2train_single(label):
    return label - 1 # uint8 trick: 0 - 1 = 255

def SemKITTI2train(label):
    if isinstance(label, list):
        return [SemKITTI2train_single(a) for a in label]
    else:
        return SemKITTI2train_single(label)

def grp_range_torch(a,dev):
    idx = torch.cumsum(a,0)
    id_arr = torch.ones(idx[-1],dtype = torch.int64,device=dev)
    id_arr[0] = 0
    id_arr[idx[:-1]] = -a[:-1]+1
    return torch.cumsum(id_arr,0)
    # generate array like [0,1,2,3,4,5,0,1,2,3,4,5,6] where each 0-n gives id to points inside the same grid

def parallel_FPS(np_cat_fea,K):
    return  nb_greedy_FPS(np_cat_fea,K)

# @nb.jit('b1[:](f4[:,:],i4)',nopython=True,cache=True)
def nb_greedy_FPS(xyz,K):
    start_element = 0
    sample_num = xyz.shape[0]
    sum_vec = np.zeros((sample_num,1),dtype = np.float32)
    xyz_sq = xyz**2
    for j in range(sample_num):
        sum_vec[j,0] = np.sum(xyz_sq[j,:])
    pairwise_distance = sum_vec + np.transpose(sum_vec) - 2*np.dot(xyz, np.transpose(xyz))

    candidates_ind = np.zeros((sample_num,),dtype = np.bool_)
    candidates_ind[start_element] = True
    remain_ind = np.ones((sample_num,),dtype = np.bool_)
    remain_ind[start_element] = False
    all_ind = np.arange(sample_num)

    for i in range(1,K):
        if i == 1:
            min_remain_pt_dis = pairwise_distance[:,start_element]
            min_remain_pt_dis = min_remain_pt_dis[remain_ind]
        else:
            cur_dis = pairwise_distance[remain_ind,:]
            cur_dis = cur_dis[:,candidates_ind]
            min_remain_pt_dis = np.zeros((cur_dis.shape[0],),dtype = np.float32)
            for j in range(cur_dis.shape[0]):
                min_remain_pt_dis[j] = np.min(cur_dis[j,:])
        next_ind_in_remain = np.argmax(min_remain_pt_dis)
        next_ind = all_ind[remain_ind][next_ind_in_remain]
        candidates_ind[next_ind] = True
        remain_ind[next_ind] = False

    return candidates_ind

def merge_ins_sem(_sem, ins, ins_classified_labels=None, ins_classified_ids=None, merge_pred_unlabeled=True, merge_few_pts_ins=True):
    sem = _sem.copy()
    ins_ids = np.unique(ins)
    for id in ins_ids:
        if id == 0: # id==0 means stuff classes
            continue
        ind = (ins == id)
        if not merge_few_pts_ins:
            if np.sum(ind) < 50:
                continue
        if ins_classified_labels is None:
            sub_sem = sem[ind]
            mode_sem_id = int(s.mode(sub_sem)[0])
            sem[ind] = mode_sem_id
        else:
            if id in ins_classified_ids:
                curr_classified_id = ins_classified_labels[ins_classified_ids==id][0]
                # mode_sem_id = ins_classified_labels[ins_classified_ids==id][0]
                if not merge_pred_unlabeled and curr_classified_id == 0: #TODO: change to use cfg
                    continue
                mode_sem_id = curr_classified_id
                sem[ind] = mode_sem_id
            else:
                sub_sem = sem[ind]
                mode_sem_id = int(s.mode(sub_sem)[0])
                sem[ind] = mode_sem_id
    return sem

def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def init_dist_slurm(batch_size, tcp_port, local_rank, backend='nccl'):
    """
    modified from https://github.com/open-mmlab/mmdetection
    Args:
        batch_size:
        tcp_port:
        backend:

    Returns:

    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

    total_gpus = dist.get_world_size()
    assert batch_size % total_gpus == 0, 'Batch size should be matched with GPUS: (%d, %d)' % (batch_size, total_gpus)
    batch_size_each_gpu = batch_size // total_gpus
    rank = dist.get_rank()
    return batch_size_each_gpu, rank


def init_dist_pytorch(batch_size, tcp_port, local_rank, backend='nccl'):    
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)
    dist.init_process_group(
        backend=backend,
        init_method='tcp://127.0.0.1:%d' % tcp_port,
        rank=local_rank,
        world_size=num_gpus
    )
    assert batch_size % num_gpus == 0, 'Batch size should be matched with GPUS: (%d, %d)' % (batch_size, num_gpus)
    batch_size_each_gpu = batch_size // num_gpus
    
    return batch_size_each_gpu, local_rank

def get_dist_info():
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def merge_evaluator(evaluator, tmp_dir, prefix=''):
    rank, world_size = get_dist_info()
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)

    dist.barrier()
    pickle.dump(evaluator, open(os.path.join(tmp_dir, '{}evaluator_part_{}.pkl'.format(prefix, rank)), 'wb'))
    dist.barrier()

    if rank != 0:
        return None

    for i in range(1, world_size):
        part_file = os.path.join(tmp_dir, '{}evaluator_part_{}.pkl'.format(prefix, i))
        evaluator.merge(pickle.load(open(part_file, 'rb')))

    return evaluator

def save_test_results(ret_dict, output_dir, batch):
    if is_nuscenes:
        assert len(ret_dict['sem_preds']) == 1
        sem_preds = ret_dict['sem_preds'][0]
        ins_preds = ret_dict['ins_preds'][0]
        label = 1000*sem_preds + ins_preds
        sample_token = batch['pcd_fname'][0]
        pcd_fname = sample_token + '_panoptic.npz'
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, pcd_fname)
        np.savez_compressed(fname, data=label.reshape(-1).astype(np.uint16))
    else:
        if class_inv_lut is None:
            raise NotImplementedError
        assert len(ret_dict['sem_preds']) == 1

        sem_preds = ret_dict['sem_preds'][0]
        ins_preds = ret_dict['ins_preds'][0]

        sem_inv = class_inv_lut[sem_preds].astype(np.uint32)
        label = sem_inv.reshape(-1, 1) + ((ins_preds.astype(np.uint32) << 16) & 0xFFFF0000).reshape(-1, 1)

        pcd_path = batch['pcd_fname'][0]
        seq = pcd_path.split('/')[-3]
        pcd_fname = pcd_path.split('/')[-1].split('.')[-2]+'.label'
        fname = os.path.join(output_dir, seq, 'predictions', pcd_fname)
        label.reshape(-1).astype(np.uint32).tofile(fname)

import open3d as o3d
np.random.seed(2121313)
color_map = np.random.rand(200, 3)
color_map[0, 0] = 0.73
color_map[0, 1] = 0.73
color_map[0, 2] = 0.73
def safe_vis(xyz, labels=None, centers=None, radius=None):
    import pdb; pdb.set_trace()
    pcobj = o3d.geometry.PointCloud()
    pcobj.points = o3d.utility.Vector3dVector(xyz)
    if labels is not None:
        colors = color_map[labels]
        pcobj.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcobj.paint_uniform_color(color_map[0])
    if centers is not None:
        pcce = o3d.geometry.PointCloud()
        pcce.points = o3d.utility.Vector3dVector(centers)
        pcce.paint_uniform_color([1, 0, 0])
        if radius is not None:
            spheres = []
            for i, r in enumerate(radius):
                mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=r)
                mesh_sphere.translate(centers[i])
                mesh_sphere.compute_vertex_normals()
                mesh_sphere.paint_uniform_color([0.86, 0.86, 0.86])
                pcd = mesh_sphere.sample_points_uniformly(number_of_points=500)
                spheres.append(pcd)
    if centers is None:
        o3d.visualization.draw_geometries([pcobj])
    elif radius is None:
        o3d.visualization.draw_geometries([pcobj, pcce])
    else:
        o3d.visualization.draw_geometries([pcobj, pcce] + spheres)
    pass


def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]],
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat


def caculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr / scale
    # must ensure pVec_Arr is also a unit vec.
    z_unit_Arr = np.array([0, 0, 1])
    z_mat = get_cross_prod_mat(z_unit_Arr)
 
    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)
 
    if np.dot(z_unit_Arr, pVec_Arr) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, pVec_Arr) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat,
                                                            z_c_vec_mat) / (1 + np.dot(z_unit_Arr, pVec_Arr))
 
    qTrans_Mat *= scale
    return qTrans_Mat


def vis_arrows(src_points, dst_points, points, gt_points, all_points):
    import pdb; pdb.set_trace()
    pcobj = o3d.geometry.PointCloud()
    pcobj.points = o3d.utility.Vector3dVector(points)
    pcobj.paint_uniform_color([0, 0, 1])
    pcobj_ = o3d.geometry.PointCloud()
    pcobj_.points = o3d.utility.Vector3dVector(gt_points)
    pcobj_.paint_uniform_color([1, 0, 0])
    pcobj_a = o3d.geometry.PointCloud()
    pcobj_a.points = o3d.utility.Vector3dVector(all_points)
    pcobj_a.paint_uniform_color([0, 1, 0])
    arrows = []
    for i, (src, dst) in enumerate(zip(src_points, dst_points)):
        vec_arr = dst - src
        mesh_arrow = o3d.geometry.TriangleMesh.create_arrow(
            cone_height=0.2 * 1 ,
            cone_radius=0.06 * 1,
            cylinder_height=0.8 * 1,
            cylinder_radius=0.04 * 1
        )
        mesh_arrow.paint_uniform_color([0, 1, 0])
        mesh_arrow.compute_vertex_normals()

        rot_mat = caculate_align_mat(vec_arr)
        mesh_arrow.rotate(rot_mat, center=np.array([0, 0, 0]))
        mesh_arrow.translate(src)
        arrows.append(mesh_arrow)
    o3d.visualization.draw_geometries(arrows+[pcobj, pcobj_, pcobj_a])
    pass
