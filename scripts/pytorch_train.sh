ngpu=${1}
bs=${2}
cfg=${3}
dir=${4}
tag=${5}
port=23456

python -m torch.distributed.launch --nproc_per_node=${ngpu} cfg_train.py \
    --tcp_port=${port} \
    --batch_size ${bs} \
    --config ${cfg} \
    --tag ${tag} \
    --log_dir ${dir} \
    --launcher pytorch