#!/bin/bash
#SBATCH -J ssf
#SBATCH --gpus-per-node 8 --nodes 1 -C "thin"
#SBATCH -t 3-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=khoche@kth.se
#SBATCH --output /proj/berzelius-2023-154/users/x_ajikh/repos/ssf/logs/slurm/%J.out
#SBATCH --error  /proj/berzelius-2023-154/users/x_ajikh/repos/ssf/logs/slurm/%J.err

cd /proj/berzelius-2023-154/users/x_ajikh/repos/ssf/

SOURCE="/proj/berzelius-2023-154/users/x_qinzh/data/av2/preprocess_v2"
DEST="/scratch/local/av2"
SUBDIRS=("sensor/train" "sensor/val")

start_time=$(date +%s)
for dir in "${SUBDIRS[@]}"; do
    mkdir -p "${DEST}/${dir}"
    find "${SOURCE}/${dir}" -type f -print0 | xargs -0 -n1 -P16 cp -t "${DEST}/${dir}" &
done
wait
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Copy ${SOURCE} to ${DEST} Total time: ${elapsed} seconds"
echo "Start training..."

# ====> paper model = seflow_official
# /proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/seflow/bin/python 1_train.py \
#     slurm_id=$SLURM_JOB_ID wandb_mode=online train_data=/scratch/local/av2/sensor/train val_data=/scratch/local/av2/sensor/val \
#     num_workers=16 model=deflow lr=2e-6 epochs=50 batch_size=20 "model.target.num_iters=2" "model.val_monitor=val/Dynamic/Mean" \
#     loss_fn=seflowLoss "add_seloss={chamfer_dis: 1.0, static_flow_loss: 1.0, dynamic_chamfer_dis: 1.0, cluster_based_pc0pc1: 1.0}"

# ====> leaderboard model = seflow_best
# /proj/berzelius-2023-154/users/x_qinzh/mambaforge/envs/seflow/bin/python 1_train.py \
#     slurm_id=$SLURM_JOB_ID wandb_mode=online train_data=/scratch/local/av2/sensor/train val_data=/scratch/local/av2/sensor/val \
#     num_workers=16 model=deflow lr=2e-4 epochs=9 batch_size=16 "model.target.num_iters=2" "model.val_monitor=val/Dynamic/Mean" \
#     loss_fn=seflowLoss "add_seloss={chamfer_dis: 1.0, static_flow_loss: 1.0, dynamic_chamfer_dis: 1.0, cluster_based_pc0pc1: 1.0}"

# ====> paper model = ssf, range = 204.8m, voxel = 0.2m
/proj/berzelius-2023-154/users/x_ajikh/.conda/envs/seflow/bin/python 1_train.py \
    slurm_id=$SLURM_JOB_ID wandb_mode=online dataset_path=/scratch/local/av2/sensor wandb_project_name=ssf \
    num_workers=16 model=ssf lr=6e-3 epochs=25 batch_size=48 loss_fn=deflowLoss val_every=5 \
    "voxel_size=[0.2, 0.2, 6]" "point_cloud_range=[-204.8, -204.8, -3, 204.8, 204.8, 3]"