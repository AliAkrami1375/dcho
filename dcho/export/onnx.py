"""ONNX export and selective quantisation.

Export removes torch from the runtime entirely. That is worth roughly two
gigabytes of install footprint, which for deployment on a small server or a
single-board computer is not a detail.

Quantisation is where care is needed, and the rule is short enough to state
plainly:

    the decoder's output projection must stay in floating point.

That layer emits the log-magnitude and phase that go straight into the
inverse STFT. Quantisation noise introduced there is not attenuated by
anything downstream - it becomes waveform noise, audible as a metallic
edge and a hiss under quiet passages. The same amount of noise inside the
transformer or the flow is absorbed by the layers after it and is
inaudible. So the body of the network is quantised to 8-bit and the output
head is not, which costs a few megabytes and a little speed in exchange for
a clean signal.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

# Node name fragments whose weights stay in floating point. Matched as
# substrings against the ONNX initializer names produced by export.
QUANTISATION_EXCLUDE = (
    "dec.conv_post",      # log-magnitude and phase projection
    "dec.ups.1",          # final upsampling stage, feeds conv_post directly
    "istft",              # the inverse DFT basis, a constant
    "pqmf",               # synthesis filter bank, a constant
    "emb_g",              # speaker table: small, and errors here shift timbre
)

OPSET = 17


class InferenceWrapper(torch.nn.Module):
    """Flattens `Synthesizer.infer` into a single-output graph for export."""

    def __init__(self, model, default_sid: int = 0):
        super().__init__()
        self.model = model
        self.default_sid = default_sid

    def forward(self, text, text_lengths, sid, noise_scale, length_scale, noise_scale_w):
        audio, *_ = self.model.infer(
            text,
            text_lengths,
            sid=sid,
            noise_scale=float(noise_scale),
            length_scale=float(length_scale),
            noise_scale_w=float(noise_scale_w),
        )
        return audio


def export_onnx(
    model,
    output_path: str | Path,
    n_speakers: int = 1,
    example_length: int = 64,
    opset: int = OPSET,
    verbose: bool = False,
) -> Path:
    """Trace the inference path to ONNX.

    `prepare_for_inference` is called first: it folds away weight
    normalisation and deletes the posterior encoder, which is roughly a
    third of the parameters and is never used outside training.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.eval()
    if hasattr(model, "prepare_for_inference"):
        model.prepare_for_inference()

    wrapper = InferenceWrapper(model).eval()

    text = torch.randint(1, 30, (1, example_length), dtype=torch.long)
    text_lengths = torch.tensor([example_length], dtype=torch.long)
    sid = torch.tensor([0], dtype=torch.long)
    scales = (torch.tensor(0.667), torch.tensor(1.0), torch.tensor(0.8))

    torch.onnx.export(
        wrapper,
        (text, text_lengths, sid, *scales),
        str(output_path),
        input_names=["text", "text_lengths", "sid", "noise_scale", "length_scale", "noise_scale_w"],
        output_names=["audio"],
        dynamic_axes={
            "text": {0: "batch", 1: "tokens"},
            "text_lengths": {0: "batch"},
            "sid": {0: "batch"},
            "audio": {0: "batch", 2: "samples"},
        },
        opset_version=opset,
        do_constant_folding=True,
        verbose=verbose,
    )
    return output_path


def quantise(
    model_path: str | Path,
    output_path: str | Path | None = None,
    exclude: tuple[str, ...] = QUANTISATION_EXCLUDE,
) -> Path:
    """Dynamic 8-bit quantisation of everything except the excluded nodes.

    Dynamic rather than static: activation ranges are computed at run time,
    so no calibration set is needed and no calibration mismatch can silently
    degrade a voice the calibration set did not represent.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    model_path = Path(model_path)
    output_path = Path(output_path) if output_path else model_path.with_suffix(".int8.onnx")

    prepped = model_path.with_suffix(".prep.onnx")
    quant_pre_process(str(model_path), str(prepped), skip_symbolic_shape=False)

    nodes_to_exclude = _matching_nodes(prepped, exclude)

    quantize_dynamic(
        model_input=str(prepped),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        nodes_to_exclude=nodes_to_exclude,
        extra_options={"MatMulConstBOnly": True},
    )
    prepped.unlink(missing_ok=True)

    manifest = output_path.with_suffix(".quant.json")
    manifest.write_text(
        json.dumps(
            {
                "source": model_path.name,
                "weight_type": "int8",
                "excluded_patterns": list(exclude),
                "excluded_nodes": nodes_to_exclude,
                "size_mb": {
                    "fp32": round(model_path.stat().st_size / 1e6, 2),
                    "int8": round(output_path.stat().st_size / 1e6, 2),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path


def _matching_nodes(model_path: Path, patterns: tuple[str, ...]) -> list[str]:
    import onnx

    model = onnx.load(str(model_path))
    names = []
    for node in model.graph.node:
        haystack = node.name + "".join(node.input)
        if any(p in haystack for p in patterns):
            names.append(node.name)
    return names


def verify(
    onnx_path: str | Path,
    torch_model,
    n_tokens: int = 48,
    tolerance: float = 0.05,
) -> dict:
    """Compare ONNX output against the torch model on the same input.

    The models are not expected to be bit-identical - both sample from the
    duration flow and the prior - so this compares length agreement and
    overall energy rather than sample-by-sample equality. It catches the
    failure that matters: a graph that exported structurally wrong.
    """
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    text = np.random.randint(1, 30, (1, n_tokens)).astype(np.int64)

    audio = session.run(
        None,
        {
            "text": text,
            "text_lengths": np.array([n_tokens], dtype=np.int64),
            "sid": np.array([0], dtype=np.int64),
            "noise_scale": np.float32(0.667),
            "length_scale": np.float32(1.0),
            "noise_scale_w": np.float32(0.8),
        },
    )[0]

    with torch.no_grad():
        reference, *_ = torch_model.infer(
            torch.from_numpy(text), torch.tensor([n_tokens]), sid=torch.tensor([0])
        )
    reference = reference.numpy()

    len_ratio = audio.shape[-1] / max(reference.shape[-1], 1)
    rms_onnx = float(np.sqrt((audio**2).mean()))
    rms_torch = float(np.sqrt((reference**2).mean()))

    return {
        "onnx_samples": int(audio.shape[-1]),
        "torch_samples": int(reference.shape[-1]),
        "length_ratio": round(len_ratio, 4),
        "rms_onnx": round(rms_onnx, 6),
        "rms_torch": round(rms_torch, 6),
        "rms_ratio": round(rms_onnx / max(rms_torch, 1e-9), 4),
        "finite": bool(np.isfinite(audio).all()),
        "ok": bool(
            np.isfinite(audio).all()
            and abs(len_ratio - 1.0) < 0.5
            and abs(rms_onnx / max(rms_torch, 1e-9) - 1.0) < tolerance * 20
        ),
    }
