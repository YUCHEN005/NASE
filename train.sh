#!/usr/bin/env bash
cmd="/home3/chenchen/research/maison2/egs/VB/slurm.pl --quiet --nodelist=node09 --gpu 2"

source activate sgmse_new

$cmd log/train.log \
python train.py \
      --base_dir data \
      --no_wandb --gpus 2 \
      --pretrain_class_model BEATs_iter3_plus_AS2M.pt \
      --inject_type addition


