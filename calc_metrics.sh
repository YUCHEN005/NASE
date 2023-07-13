#!/usr/bin/env bash
cmd="/home3/chenchen/research/maison2/egs/VB/slurm.pl --quiet --nodelist=node01 --gpu 1 --num-threads 2"

source activate sgmse_new

$cmd log/calc_metrics.log \
python calc_metrics.py \
      --test_dir data/test \
      --test_set noisy \
      --enhanced_dir enhanced
