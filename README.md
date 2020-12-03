## Preliminary code for reviewers only


## Prerequisites

* Python 3.6
* Pytorch 1.3
* CUDA 9.0


## Training

```bash
python ./tools/train.py --dataset refcoco --splitBy unc --exp_id 1
```

## Evaluation

```bash
python ./eval.py --dataset refcoco --splitBy unc --split val --id 1
```

