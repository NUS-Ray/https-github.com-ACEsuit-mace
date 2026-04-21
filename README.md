# MACE install

This repository documents how to install [ACEsuit/mace](https://github.com/ACEsuit/mace) (the
MACE machine-learning interatomic potential library) in this workspace.

MACE has been installed from source into a Python virtual environment at `./.venv`
and verified to import correctly and expose its CLI entry points.

## What was installed

- Python 3.12 virtual environment at `./.venv`
- PyTorch 2.11.0 (CPU build — no CUDA GPU is available in this environment)
- `mace-torch` 0.3.15, installed from source by cloning
  [ACEsuit/mace](https://github.com/ACEsuit/mace) into `./mace` and running
  `pip install ./mace`
- Transitive deps pulled in by MACE: `e3nn`, `ase`, `matscipy`, `numpy`, `scipy`,
  `torch-ema`, `torchmetrics`, `h5py`, `prettytable`, `opt-einsum-fx`, `configargparse`,
  `GitPython`, `matplotlib`, `pandas`, `lmdb`, `orjson`, `tqdm`, `PyYAML`, …

The local clone in `./mace/` and the `./.venv/` directory are intentionally excluded
from version control via `.gitignore`.

## Reproducing the install

Run these commands from the repository root:

```bash
sudo apt-get update
sudo apt-get install -y python3.12-venv

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

pip install --index-url https://download.pytorch.org/whl/cpu torch

git clone https://github.com/ACEsuit/mace.git
pip install ./mace
```

If you have a CUDA-capable GPU, install the CUDA build of PyTorch instead —
pick the right command for your CUDA version from
<https://pytorch.org/get-started/locally/>, for example:

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch
```

## Verifying the install

```bash
source .venv/bin/activate

python -c "import mace, mace.calculators, mace.modules, mace.cli; print('mace', mace.__version__)"

mace_run_train --help | head
```

Expected output includes `mace 0.3.15` and the `mace_run_train` argparse help.

The following CLI entry points are installed into `./.venv/bin/`:

- `mace_run_train`
- `mace_eval_configs`
- `mace_create_lammps_model`
- `mace_active_learning_md`
- `mace_convert_device`
- `mace_finetuning`
- `mace_plot_train`
- `mace_prepare_data`
- `mace_select_head`
- `mace_update_input_checkpoint`

## Using MACE

With the venv activated, see the upstream
[MACE README](https://github.com/ACEsuit/mace#usage) and
[docs](https://mace-docs.readthedocs.io/) for training/evaluation commands and
foundation-model usage.

Minimal Python smoke test (CPU):

```python
import torch
from e3nn import o3
from mace.modules import MACE
from mace.modules.blocks import (
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
)

model = MACE(
    r_max=5.0,
    num_bessel=8,
    num_polynomial_cutoff=5,
    max_ell=2,
    interaction_cls=RealAgnosticResidualInteractionBlock,
    interaction_cls_first=RealAgnosticInteractionBlock,
    num_interactions=2,
    num_elements=1,
    hidden_irreps=o3.Irreps("8x0e"),
    MLP_irreps=o3.Irreps("16x0e"),
    gate=torch.nn.functional.silu,
    atomic_energies=torch.tensor([0.0]),
    avg_num_neighbors=8,
    atomic_numbers=[1],
    correlation=2,
)
print(sum(p.numel() for p in model.parameters()), "parameters")
```

For GPU accelerated foundation-model use (requires CUDA):

```python
from mace.calculators import mace_mp
from ase import build

atoms = build.molecule("H2O")
calc = mace_mp(model="medium", dispersion=False, default_dtype="float32", device="cuda")
atoms.calc = calc
print(atoms.get_potential_energy())
```
