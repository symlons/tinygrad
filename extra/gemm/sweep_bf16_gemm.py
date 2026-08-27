"""Reproducible custom/cuBLAS/cuDNN BF16 GEMM sweep for the A100.

Shape tuples use (M, N, K), while reports show the multiplication as
MxK @ KxN to avoid the historically ambiguous M/K/N table ordering.
"""
from __future__ import annotations

import argparse, json, os, re, subprocess, sys, time
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Shape:
  name: str
  m: int
  n: int
  k: int
  reason: str


DOCUMENTED = (
  Shape("square-1024", 1024, 1024, 1024, "small square / fixed-cost floor"),
  Shape("square-2048", 2048, 2048, 2048, "medium square"),
  Shape("square-4096", 4096, 4096, 4096, "large square"),
  Shape("square-8192", 8192, 8192, 8192, "large square / peak throughput"),
  Shape("wide-14336", 4096, 14336, 4096, "production wide projection"),
  Shape("deep-14336", 4096, 4096, 14336, "production deep projection"),
  Shape("wide-28672", 8192, 28672, 8192, "production wide projection"),
  Shape("deep-28672", 8192, 8192, 28672, "production deep projection"),
  Shape("short-m-2048", 2048, 8192, 8192, "rectangular M"),
  Shape("short-n-2048", 8192, 2048, 8192, "rectangular N"),
  Shape("thin-k-128", 8192, 8192, 128, "persistent-kernel target"),
  Shape("thin-n-128", 8192, 128, 8192, "four-warp 128x128 target"),
  Shape("thin-m-256", 256, 8192, 8192, "occupancy-limited M"),
)


ADDED = (
  Shape("square-512", 512, 512, 512, "fixed launch/epilogue cost below 1024"),
  Shape("k-transition-256", 8192, 8192, 256, "persistent path scaling"),
  Shape("k-transition-512", 8192, 8192, 512, "current persistent threshold"),
  Shape("k-transition-768", 8192, 8192, 768, "first point above persistent threshold"),
  Shape("k-transition-1024", 8192, 8192, 1024, "flat-kernel recovery after threshold"),
  Shape("thin-n-256", 8192, 256, 8192, "N tile-count scaling"),
  Shape("thin-n-512", 8192, 512, 8192, "N tile-count scaling"),
  Shape("thin-m-512", 512, 8192, 8192, "M tile-count scaling"),
  Shape("thin-m-1024", 1024, 8192, 8192, "M tile-count scaling"),
  Shape("cta-boundary-104", 3328, 1024, 8192, "104 output CTAs, just below 108 SMs"),
  Shape("cta-boundary-112", 3584, 1024, 8192, "112 output CTAs, just above 108 SMs"),
  Shape("llm-wide-11008", 4096, 11008, 4096, "common LLM projection width"),
  Shape("llm-deep-11008", 4096, 4096, 11008, "transpose of common LLM projection"),
)

LINE_RE = re.compile(r"^(tinygrad NV \w+|cuBLAS GemmEx|cuDNN graph matmul)\s+([0-9.]+) TFLOPS\s+([0-9.]+) ms", re.M)


def timing(shape:Shape) -> tuple[int, int]:
  ops = 2 * shape.m * shape.n * shape.k
  if ops < 20_000_000_000: return 5000, 300
  if ops < 300_000_000_000: return 3000, 200
  return 500, 100


def run(shape:Shape, backend:str) -> tuple[dict[str, dict[str, float]], str]:
  warmup, iters = timing(shape)
  env = os.environ.copy() | {"DEV":"NV" if backend == "custom" else "CUDA", "M":str(shape.m), "N":str(shape.n), "K":str(shape.k),
                             "WARMUP":str(warmup), "ITERS":str(iters), "COOLDOWN":"0"}
  proc = subprocess.run([sys.executable, "-m", "extra.gemm.benchmark_bf16_gemm"], cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
  found = {name:{"tflops":float(tflops), "ms":float(ms)} for name, tflops, ms in LINE_RE.findall(proc.stdout)}
  if proc.returncode or not found: raise RuntimeError(f"{backend} failed for {shape.name}:\n{proc.stdout}")
  return found, proc.stdout


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--set", choices=("documented", "added", "all"), default="all")
  parser.add_argument("--backend", choices=("custom", "vendor", "all"), default="all")
  parser.add_argument("--json", dest="json_path")
  args = parser.parse_args()
  shapes = DOCUMENTED if args.set == "documented" else ADDED if args.set == "added" else DOCUMENTED + ADDED
  backends = ("custom", "vendor") if args.backend == "all" else (args.backend,)
  results = []
  for idx, shape in enumerate(shapes, 1):
    row:dict = asdict(shape) | {"shape":f"{shape.m}x{shape.k} @ {shape.k}x{shape.n}", "results":{}}
    print(f"[{idx}/{len(shapes)}] {shape.name}: {row['shape']}", flush=True)
    for backend in backends:
      values, output = run(shape, backend)
      row["results"].update(values)
      print("  " + " | ".join(f"{name} {value['tflops']:.2f} TFLOPS" for name, value in values.items()), flush=True)
      row.setdefault("raw", {})[backend] = output
    results.append(row)
  payload = {"timestamp":time.strftime("%Y-%m-%dT%H:%M:%S%z"), "gpu":"NVIDIA A100-SXM4-40GB", "results":results}
  if args.json_path:
    with open(args.json_path, "w") as f: json.dump(payload, f, indent=2)


if __name__ == "__main__": main()
