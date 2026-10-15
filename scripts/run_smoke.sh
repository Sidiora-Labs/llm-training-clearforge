#!/usr/bin/env bash
# Execute from the repository root after installing Clearforge.
# Output directories must be new. This fixture checks mechanics, not chat quality.
set -euo pipefail
clearforge data prepare --source examples/pretrain.jsonl --license MIT \
  --language en --mode pretrain --output data/smoke-prepared --validation-fraction 0.2
clearforge tokenizer train --input data/smoke-prepared/train.jsonl \
  --output data/smoke-tokenizer --vocab-size 512
clearforge data pack --input data/smoke-prepared --tokenizer data/smoke-tokenizer \
  --output data/smoke-packed --seq-length 64 --mode pretrain
clearforge train configs/smoke.toml --stop-after-steps 4
clearforge train configs/smoke.toml --resume runs/smoke-pretrain/latest.pt
clearforge data prepare --source examples/chat.jsonl --license MIT \
  --language en --mode sft --output data/smoke-sft-prepared --validation-fraction 0.2
clearforge data pack --input data/smoke-sft-prepared --tokenizer data/smoke-tokenizer \
  --output data/smoke-sft-packed --seq-length 64 --mode sft
clearforge train configs/smoke-sft.toml
clearforge evaluate configs/smoke-sft.toml --checkpoint runs/smoke-sft/latest.pt
clearforge export configs/smoke-sft.toml --checkpoint runs/smoke-sft/latest.pt \
  --output runs/smoke-export
clearforge chat --model runs/smoke-export --prompt "What is a library?" \
  --max-new-tokens 8 --device cpu
