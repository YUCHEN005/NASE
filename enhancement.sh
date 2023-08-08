#!/usr/bin/env bash
cmd="/home3/chenchen/research/maison2/egs/VB/slurm.pl --quiet --nodelist=node01 --gpu 1 --num-threads 2"

source activate sgmse_new

$cmd log/enhancement.log \
python enhancement.py \
      --test_dir data/test \
      --test_set noisy \
      --enhanced_dir enhanced \
      --ckpt logs/mtl_with_beats/epoch_97_pesq_3.01.ckpt \
      --pretrain_class_model <your_path>/BEATs_iter3_plus_AS2M.pt

