# MACE install

This repository documents how to install [ACEsuit/mace](https://github.com/ACEsuit/mace) (the
MACE machine-learning interatomic potential library) in this workspace.

MACE has been installed from source into a Python virtual environment at `./.venv`
and verified to import correctly and expose its CLI entry points.

## What was installed

- Python 3.12 virtual environment at `./.venv`
- PyTorch 2.11.0 (CPU build — no CUDA GPU is available in this environment).
  This satisfies MACE's requirement of PyTorch >= 1.12 and avoids the known-bad
  versions: `2.1.x` (float64 training not supported) and `2.4.1` (not supported
  by MACE). Float64 training is supported (verified: building a MACE model with
  `torch.set_default_dtype(torch.float64)` produces float64 parameters).
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

### PyTorch version constraints (from the upstream MACE README)

- PyTorch **>= 1.12**.
- PyTorch **2.1.x** is not recommended — training with `float64` is not
  supported on 2.1. Use 2.2 or later for `float64` training.
- PyTorch **2.4.1** is **not supported** by MACE. Pick a different patch
  version (for example 2.4.0 or 2.5+).

The pin-free command above installs the latest stable PyTorch, which satisfies
all of these constraints. If you need to pin to a known-good version explicitly
you can, for example, do:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,!=2.4.1"
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

## Web UI (`app.py`)

A minimal [Gradio](https://gradio.app/) web UI is included in `app.py` so you can
drive the MACE install from a browser instead of a Python REPL.

Install the extra dependency and launch:

```bash
source .venv/bin/activate
pip install -r requirements-ui.txt
python app.py
```

Then open the URL that Gradio prints (default <http://127.0.0.1:7860>).

The UI has four tabs:

1. **Environment** — shows the installed versions of Python, PyTorch, MACE,
   e3nn, ASE and NumPy, and whether CUDA is available.
2. **Build MACE model** — lets you pick `r_max`, `num_channels`, `max_L`,
   `num_interactions`, `correlation`, `num_bessel`, `num_polynomial_cutoff`,
   dtype (`float32`/`float64`) and a list of atomic numbers, then builds a
   random-initialised `mace.modules.MACE` model and reports its parameter
   count and architecture. Useful for sanity-checking the install and
   exploring model-size trade-offs.
3. **Evaluate checkpoint** — point it at a MACE `.model` checkpoint (for
   example any file from the [mace-foundations
   releases](https://github.com/ACEsuit/mace-foundations/releases)) and
   enter either an ASE formula (`H2O`, `CH4`, …) or extended-XYZ text; the
   app loads a `MACECalculator` and returns the potential energy and
   per-atom forces. Set `device=cuda` if you have a GPU.
4. **MACE CLI** — runs whitelisted `mace_*` CLI commands
   (`mace_run_train --help`, `mace_eval_configs --help`, …) and shows their
   stdout/stderr. Only commands starting with `mace_` are allowed.

Environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `MACE_UI_HOST` | `127.0.0.1` | Address to bind the Gradio server to. Set to `0.0.0.0` to expose on the LAN. |
| `MACE_UI_PORT` | `7860` | Port for the Gradio server. |
| `MACE_UI_SHARE` | `0` | Set to `1` to create a public `*.gradio.live` share link. |

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
