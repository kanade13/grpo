# GRPO environment requirements

Recommended GPU target: NVIDIA RTX 3060
Recommended Python: 3.10
Recommended CUDA runtime package: `pytorch-cuda=12.1`

## Conda base environment

```bash
conda create -n grpo python=3.10 pip -y
conda activate grpo
```

## PyTorch with CUDA 12.1

Preferred conda install:

```bash
conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1
```

If conda resolution is unstable on your machine, use pip wheels instead:

```bash
python -m pip install --upgrade --force-reinstall --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

## Python packages

```bash
export PYTHONNOUSERSITE=1
python -m pip install -r /mnt/g/grpo/requirements-grpo.txt
```

`PYTHONNOUSERSITE=1` is recommended on this machine because the user site-packages under `~/.local` can leak into conda environments.

## Verification

```bash
export PYTHONNOUSERSITE=1
python - <<'PY'
import torch, transformers, datasets, accelerate, trl, peft
print('torch', torch.__version__)
print('cuda', torch.version.cuda)
print('cuda_available', torch.cuda.is_available())
print('gpu_count', torch.cuda.device_count())
print('transformers', transformers.__version__)
print('datasets', datasets.__version__)
print('accelerate', accelerate.__version__)
print('trl', trl.__version__)
print('peft', peft.__version__)
PY
```
