# Noise-aware Speech Enhancement using Diffusion Probabilistic Model

<p align="center">  <img src="https://github.com/YUCHEN005/NASE/blob/master/nase.png" height ="230"> </p>

This repository contains the official PyTorch implementations for our paper:

- Yuchen Hu, Chen Chen, Ruizhe Li, Qiushi Zhu, Eng Siong Chng. [*"Noise-aware Speech Enhancement using Diffusion Probabilistic Model"*](https://arxiv.org/abs/2307.08029).

Our code is based on prior work [SGMSE+](https://github.com/sp-uhh/sgmse).


## Installation

- Create a new virtual environment with Python 3.8 (we have not tested other Python versions, but they may work).
- Install the package dependencies via `pip install -r requirements.txt`.
- If using W&B logging (default):
    - Set up a [wandb.ai](https://wandb.ai/) account
    - Log in via `wandb login` before running our code.
- If not using W&B logging:
    - Pass the option `--no_wandb` to `train.py`.
    - Your logs will be stored as local TensorBoard logs. Run `tensorboard --logdir logs/` to see them.


## Pretrained checkpoints

- We release [pretrained checkpoint](https://drive.google.com/drive/folders/1q25IOSR5Xd-5Kv13PfOhVJvMhgssTJPh?usp=sharing) for the model trained on VoiceBank-DEMAND, as in the paper.
- We also provide [testing samples](https://drive.google.com/drive/folders/18wCBq2I_W2sTdQL1OkBHU0KnLqLDHFjd?usp=sharing) before and after NASE processing for comparison.

Usage:
- For resuming training, you can use the `--resume_from_checkpoint` option of `train.py`.
- For evaluating these checkpoints, use the `--ckpt` option of `enhancement.py` (see section **Evaluation** below).


## Training

Training is done by executing `train.py`. A minimal running example with default settings can be run with:

```bash
python train.py --base_dir <your_base_dir> --inject_type <inject_type> --pretrain_class_model <pretrained_beats>
```

where `your_base_dir` should be a path to a folder containing subdirectories `train/` and `valid/` (optionally `test/` as well). Each subdirectory must itself have two subdirectories `clean/` and `noisy/`, with the same filenames present in both. We currently only support training with `.wav` files.
`inject_type` should be chosen from ["addition", "concat", "cross-attention"].
`pretrained_beats` should be the path to pre-trained [BEATs](https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D).

The full command is also included in `train.sh`.
To see all available training options, run `python train.py --help`.


## Evaluation

To evaluate on a test set, run
```bash
python enhancement.py --test_dir <your_test_dir> --enhanced_dir <your_enhanced_dir> --ckpt <path_to_model_checkpoint> --pretrain_class_model <pretrained_beats>
```

to generate the enhanced .wav files, and subsequently run

```bash
python calc_metrics.py --test_dir <your_test_dir> --enhanced_dir <your_enhanced_dir>
```

to calculate and output the instrumental metrics.

Both scripts should receive the same `--test_dir` and `--enhanced_dir` parameters. 
The `--cpkt` parameter of `enhancement.py` should be the path to a trained model checkpoint, as stored by the logger in `logs/`.
The `--pretrain_class_model` should be the path to pre-trained [BEATs](https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D).

You may refer to our full commands included in `enhancement.sh` and `calc_metrics.sh`. 

## Citations

We kindly hope you can cite our paper in your publication when using our research or code:
```bib
@inproceedings{hu2024noise,
  title={Noise-aware Speech Enhancement using Diffusion Probabilistic Model}, 
  author={Hu, Yuchen and Chen, Chen and Li, Ruizhe and Zhu, Qiushi and Chng, Eng Siong},
  booktitle={INTERSPEECH},
  year={2024}
}
```