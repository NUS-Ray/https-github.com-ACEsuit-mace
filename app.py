"""
Simple Gradio UI for the MACE installation in this workspace.

Run (with the workspace venv activated):

    source .venv/bin/activate
    python app.py

Then open the printed URL (usually http://127.0.0.1:7860) in your browser.
"""

from __future__ import annotations

import io
import os
import platform
import sys
import traceback
from contextlib import redirect_stdout
from typing import Optional

import gradio as gr


def _env_info() -> str:
    lines: list[str] = []
    lines.append(f"Python       : {platform.python_version()} ({sys.executable})")
    lines.append(f"Platform     : {platform.platform()}")

    try:
        import torch

        lines.append(f"PyTorch      : {torch.__version__}")
        lines.append(f"CUDA avail.  : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            lines.append(f"CUDA device  : {torch.cuda.get_device_name(0)}")
        lines.append(f"Default dtype: {torch.get_default_dtype()}")
    except Exception as e:  # pragma: no cover
        lines.append(f"PyTorch      : NOT INSTALLED ({e})")

    try:
        import mace

        lines.append(f"MACE         : {getattr(mace, '__version__', 'installed')}")
    except Exception as e:  # pragma: no cover
        lines.append(f"MACE         : NOT INSTALLED ({e})")

    try:
        import e3nn

        lines.append(f"e3nn         : {e3nn.__version__}")
    except Exception as e:  # pragma: no cover
        lines.append(f"e3nn         : NOT INSTALLED ({e})")

    try:
        import ase

        lines.append(f"ASE          : {ase.__version__}")
    except Exception as e:  # pragma: no cover
        lines.append(f"ASE          : NOT INSTALLED ({e})")

    try:
        import numpy as np

        lines.append(f"NumPy        : {np.__version__}")
    except Exception as e:  # pragma: no cover
        lines.append(f"NumPy        : NOT INSTALLED ({e})")

    return "\n".join(lines)


def build_mace_model(
    r_max: float,
    num_channels: int,
    max_L: int,
    num_interactions: int,
    correlation: int,
    num_bessel: int,
    num_polynomial_cutoff: int,
    dtype: str,
    atomic_numbers_str: str,
) -> tuple[str, str]:
    """Build a (random-initialised) MACE model and return summary info."""
    try:
        import torch
        from e3nn import o3

        import mace  # noqa: F401  (ensures env var for e3nn weights-only load)
        from mace.modules import MACE
        from mace.modules.blocks import (
            RealAgnosticInteractionBlock,
            RealAgnosticResidualInteractionBlock,
        )

        torch_dtype = torch.float64 if dtype == "float64" else torch.float32
        torch.set_default_dtype(torch_dtype)

        atomic_numbers = [
            int(tok) for tok in atomic_numbers_str.replace(",", " ").split() if tok
        ]
        if not atomic_numbers:
            atomic_numbers = [1]

        hidden_irreps = o3.Irreps(
            " + ".join(
                f"{num_channels}x{ell}{'e' if ell % 2 == 0 else 'o'}"
                for ell in range(max_L + 1)
            )
        )
        mlp_irreps = o3.Irreps(f"{max(num_channels // 8, 8)}x0e")

        model = MACE(
            r_max=float(r_max),
            num_bessel=int(num_bessel),
            num_polynomial_cutoff=int(num_polynomial_cutoff),
            max_ell=max(2, max_L),
            interaction_cls=RealAgnosticResidualInteractionBlock,
            interaction_cls_first=RealAgnosticInteractionBlock,
            num_interactions=int(num_interactions),
            num_elements=len(atomic_numbers),
            hidden_irreps=hidden_irreps,
            MLP_irreps=mlp_irreps,
            gate=torch.nn.functional.silu,
            atomic_energies=torch.zeros(len(atomic_numbers), dtype=torch_dtype),
            avg_num_neighbors=8,
            atomic_numbers=atomic_numbers,
            correlation=int(correlation),
        )

        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_dtypes = {str(p.dtype) for p in model.parameters()}

        summary_lines = [
            f"Built MACE model (random-initialised).",
            "",
            f"hidden_irreps   : {hidden_irreps}",
            f"MLP_irreps      : {mlp_irreps}",
            f"r_max           : {r_max}",
            f"num_interactions: {num_interactions}",
            f"correlation     : {correlation}",
            f"atomic_numbers  : {atomic_numbers}",
            f"dtype           : {dtype}",
            f"param dtypes    : {sorted(param_dtypes)}",
            f"# parameters    : {n_params:,}",
            f"# trainable     : {n_trainable:,}",
        ]
        return "\n".join(summary_lines), repr(model)
    except Exception:
        return "ERROR:\n" + traceback.format_exc(), ""


def evaluate_with_checkpoint(
    model_path: str,
    formula_or_xyz: str,
    default_dtype: str,
    device: str,
) -> str:
    """Load a pretrained MACE .model checkpoint and evaluate it on a molecule.

    ``formula_or_xyz`` can be either:
      * an ASE-buildable formula like ``H2O`` or ``CH4``
      * extended XYZ text (first line is atom count, second is a comment)
    """
    try:
        if not model_path or not os.path.isfile(model_path):
            return (
                f"Model file not found: {model_path!r}\n"
                "Point this to a MACE .model checkpoint, e.g. one of the files\n"
                "listed at https://github.com/ACEsuit/mace-foundations/releases."
            )

        import numpy as np
        import torch
        from ase import build, io as ase_io

        import mace  # noqa: F401
        from mace.calculators import MACECalculator

        if device == "cuda" and not torch.cuda.is_available():
            return "CUDA requested but not available in this environment. Pick 'cpu'."

        text = formula_or_xyz.strip()
        if not text:
            return "Please provide either a chemical formula (e.g. H2O) or XYZ text."

        atoms = None
        if "\n" not in text and len(text) < 40:
            try:
                atoms = build.molecule(text)
            except Exception:
                atoms = None

        if atoms is None:
            with io.StringIO(text) as buf:
                atoms = ase_io.read(buf, format="extxyz")

        calc = MACECalculator(
            model_paths=model_path, device=device, default_dtype=default_dtype
        )
        atoms.calc = calc

        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces())

        lines = [
            f"Loaded checkpoint : {model_path}",
            f"Device            : {device}",
            f"Default dtype     : {default_dtype}",
            f"Atoms             : {atoms.get_chemical_formula()} (N={len(atoms)})",
            f"Energy [eV]       : {energy:.6f}",
            f"max |F| [eV/Å]    : {np.max(np.linalg.norm(forces, axis=1)):.6f}",
            "",
            "Forces [eV/Å]:",
        ]
        for sym, f in zip(atoms.get_chemical_symbols(), forces):
            lines.append(f"  {sym:>2s}  {f[0]:+.6f}  {f[1]:+.6f}  {f[2]:+.6f}")
        return "\n".join(lines)
    except Exception:
        return "ERROR:\n" + traceback.format_exc()


def run_cli(cmd: str) -> str:
    """Run a MACE CLI command (``mace_run_train --help`` etc.) and return output.

    Only commands starting with ``mace_`` are allowed, to keep the UI focused.
    """
    import shlex
    import subprocess

    cmd = cmd.strip()
    if not cmd:
        return "Enter a command like:  mace_run_train --help"
    try:
        tokens = shlex.split(cmd)
    except ValueError as e:
        return f"Could not parse command: {e}"
    if not tokens[0].startswith("mace_"):
        return (
            "For safety, only commands that start with 'mace_' are allowed from "
            "this UI (e.g. mace_run_train, mace_eval_configs)."
        )
    try:
        proc = subprocess.run(
            tokens, capture_output=True, text=True, timeout=120, check=False
        )
        parts = [f"$ {cmd}", f"exit code: {proc.returncode}"]
        if proc.stdout:
            parts += ["", "--- stdout ---", proc.stdout.rstrip()]
        if proc.stderr:
            parts += ["", "--- stderr ---", proc.stderr.rstrip()]
        return "\n".join(parts)
    except subprocess.TimeoutExpired:
        return "Command timed out after 120 seconds."
    except FileNotFoundError:
        return (
            f"Command not found: {tokens[0]!r}. Make sure the venv is activated "
            "and mace is installed."
        )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="MACE UI") as demo:
        gr.Markdown(
            "# MACE UI\n"
            "A minimal web UI around the MACE installation in this workspace. "
            "Use the tabs below to inspect the environment, build a MACE model, "
            "evaluate a pretrained checkpoint on a molecule, or run MACE CLI "
            "commands."
        )

        with gr.Tab("Environment"):
            env_out = gr.Textbox(
                label="Installed versions",
                value=_env_info(),
                lines=10,
                interactive=False,
            )
            refresh = gr.Button("Refresh")
            refresh.click(lambda: _env_info(), outputs=env_out)

        with gr.Tab("Build MACE model"):
            gr.Markdown(
                "Build a **random-initialised** MACE model with the given hyper-"
                "parameters and show its size. This does not train or make "
                "physical predictions — it is a quick way to confirm the install "
                "works and to explore model-size trade-offs."
            )
            with gr.Row():
                r_max = gr.Slider(2.0, 8.0, value=5.0, step=0.1, label="r_max (Å)")
                num_channels = gr.Slider(
                    4, 256, value=32, step=4, label="num_channels"
                )
                max_L = gr.Slider(0, 3, value=1, step=1, label="max_L")
            with gr.Row():
                num_interactions = gr.Slider(
                    1, 4, value=2, step=1, label="num_interactions"
                )
                correlation = gr.Slider(1, 4, value=2, step=1, label="correlation")
                num_bessel = gr.Slider(4, 16, value=8, step=1, label="num_bessel")
                num_polynomial_cutoff = gr.Slider(
                    2, 8, value=5, step=1, label="num_polynomial_cutoff"
                )
            with gr.Row():
                dtype = gr.Radio(
                    choices=["float32", "float64"], value="float64", label="dtype"
                )
                atomic_numbers_str = gr.Textbox(
                    value="1 6 7 8",
                    label="atomic numbers (space or comma separated)",
                )
            build_btn = gr.Button("Build model", variant="primary")
            summary = gr.Textbox(label="Summary", lines=14)
            arch = gr.Textbox(
                label="Model architecture (repr)",
                lines=18,
            )
            build_btn.click(
                build_mace_model,
                inputs=[
                    r_max,
                    num_channels,
                    max_L,
                    num_interactions,
                    correlation,
                    num_bessel,
                    num_polynomial_cutoff,
                    dtype,
                    atomic_numbers_str,
                ],
                outputs=[summary, arch],
            )

        with gr.Tab("Evaluate checkpoint"):
            gr.Markdown(
                "Load a pretrained MACE `.model` checkpoint (for example, a "
                "[mace-foundations release](https://github.com/ACEsuit/mace-foundations/releases)) "
                "and evaluate it on a molecule. The input can be either a "
                "chemical formula that ASE can build (`H2O`, `CH4`, …) or "
                "extended-XYZ text."
            )
            model_path = gr.Textbox(
                label="Path to MACE .model checkpoint",
                placeholder="/absolute/path/to/your_model.model",
            )
            formula_or_xyz = gr.Textbox(
                label="ASE formula or XYZ",
                value="H2O",
                lines=6,
            )
            with gr.Row():
                default_dtype = gr.Radio(
                    choices=["float32", "float64"],
                    value="float64",
                    label="default_dtype",
                )
                device = gr.Radio(
                    choices=["cpu", "cuda"], value="cpu", label="device"
                )
            eval_btn = gr.Button("Evaluate", variant="primary")
            eval_out = gr.Textbox(
                label="Result", lines=18
            )
            eval_btn.click(
                evaluate_with_checkpoint,
                inputs=[model_path, formula_or_xyz, default_dtype, device],
                outputs=eval_out,
            )

        with gr.Tab("MACE CLI"):
            gr.Markdown(
                "Run a MACE CLI command (only commands starting with `mace_` "
                "are allowed). Useful for `mace_run_train --help`, "
                "`mace_eval_configs --help`, etc."
            )
            cli_in = gr.Textbox(
                label="Command",
                value="mace_run_train --help",
                lines=2,
            )
            cli_btn = gr.Button("Run", variant="primary")
            cli_out = gr.Textbox(label="Output", lines=20)
            cli_btn.click(run_cli, inputs=cli_in, outputs=cli_out)

    return demo


if __name__ == "__main__":
    server_name = os.environ.get("MACE_UI_HOST", "127.0.0.1")
    server_port = int(os.environ.get("MACE_UI_PORT", "7860"))
    share = os.environ.get("MACE_UI_SHARE", "0") == "1"
    demo = build_ui()
    demo.launch(server_name=server_name, server_port=server_port, share=share)
