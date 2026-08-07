"""Command line interface."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def cmd_say(args) -> int:
    from .infer.engine import TTS, SynthesisOptions

    text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
    if not text:
        print("nothing to say", file=sys.stderr)
        return 1

    tts = TTS(args.model, num_threads=args.threads)
    opt = SynthesisOptions(speaker=args.speaker, speed=args.speed)
    started = time.perf_counter()
    audio = tts(text, opt)
    elapsed = time.perf_counter() - started

    from .infer.engine import write_wav

    write_wav(args.out, audio, tts.sample_rate)
    seconds = len(audio) / tts.sample_rate
    print(f"{args.out}  {seconds:.2f}s audio in {elapsed:.2f}s  (RTF {elapsed/max(seconds,1e-9):.4f})")
    return 0


def cmd_phonemize(args) -> int:
    from .text.frontend import Frontend
    from .text.phonemes import to_ipa

    fe = Frontend()
    text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
    for line in text.splitlines():
        if not line.strip():
            continue
        out = fe(line)
        print(out.phonemes)
        if args.ipa:
            print("  " + to_ipa(out.phonemes))
        if args.verbose:
            print(f"  normalised: {out.normalized}")
            marked = [t + ("+" if e else "") for t, e in zip(out.tokens, out.ezafe)]
            print(f"  tokens    : {' '.join(marked)}")
    return 0


def cmd_normalize(args) -> int:
    from .text.normalizer import normalize

    text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
    for line in text.splitlines():
        print(normalize(line))
    return 0


def cmd_bench(args) -> int:
    """Measure the real-time factor, which is the product's binding constraint."""
    import numpy as np

    from .infer.engine import TTS, SynthesisOptions

    sample = args.text or (
        "زبان فارسی یکی از زبان‌های کهن جهان است و پیشینه‌ای بیش از هزار سال دارد. "
        "این متن برای سنجش سرعت تولید گفتار به کار می‌رود."
    )
    for threads in args.threads:
        tts = TTS(args.model, num_threads=threads)
        opt = SynthesisOptions()
        tts(sample, opt)  # warm up
        times, seconds = [], 0.0
        for _ in range(args.runs):
            t0 = time.perf_counter()
            audio = tts(sample, opt)
            times.append(time.perf_counter() - t0)
            seconds = len(audio) / tts.sample_rate
        median = float(np.median(times))
        print(f"threads={threads}  audio={seconds:.2f}s  synth={median:.3f}s  RTF={median/seconds:.4f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dcho", description="Persian text to speech")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("say", help="synthesise text to a wav file")
    p.add_argument("text", nargs="?", default=None)
    p.add_argument("-f", "--file")
    p.add_argument("-o", "--out", default="out.wav")
    p.add_argument("-m", "--model", default="models/dcho-base.int8.onnx")
    p.add_argument("-s", "--speaker", type=int, default=0)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--threads", type=int, default=2)
    p.set_defaults(func=cmd_say)

    p = sub.add_parser("phonemize", help="show the phoneme sequence for text")
    p.add_argument("text", nargs="?", default=None)
    p.add_argument("-f", "--file")
    p.add_argument("--ipa", action="store_true", help="also print IPA")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_phonemize)

    p = sub.add_parser("normalize", help="show normalised text")
    p.add_argument("text", nargs="?", default=None)
    p.add_argument("-f", "--file")
    p.set_defaults(func=cmd_normalize)

    p = sub.add_parser("bench", help="measure the real-time factor")
    p.add_argument("text", nargs="?", default=None)
    p.add_argument("-m", "--model", default="models/dcho-base.int8.onnx")
    p.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--runs", type=int, default=5)
    p.set_defaults(func=cmd_bench)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
