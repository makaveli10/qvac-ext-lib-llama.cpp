#!/bin/bash
# Smoke test for tensor parallelism with flash attention off (inference, LoRA inference, LoRA training, full training).
# usage: scripts/tp-check.sh <build-dir> <f32-model.gguf> [q-model.gguf] [text-file]
# env: TP_DEVS (default CUDA0,CUDA1), TP_RPC (e.g. host:port,host:port to add --rpc)
set -u
BUILD=${1:?build dir}
MODEL_F32=${2:?f32 model}
MODEL_Q=${3:-$MODEL_F32}
TEXT=${4:-docs/build.md}
DEVS=${TP_DEVS:-CUDA0,CUDA1}
RPC=${TP_RPC:+--rpc $TP_RPC}
OUT=tp-check-out
mkdir -p $OUT

TP="$RPC -dev $DEVS -sm tensor -ngl 99"
REF="-dev ${DEVS%%,*} -ngl 99"

ppl() { # name, args...
    local name=$1; shift
    "$BUILD/bin/llama-perplexity" -m "$MODEL_Q" -f "$TEXT" -c 512 --chunks 8 -b 512 "$@" > $OUT/ppl_$name.log 2>&1
    local rc=$?
    printf "%-22s rc=%-3s %s\n" "$name" "$rc" "$(grep -E 'Final estimate|ASSERT|unsupported|error' $OUT/ppl_$name.log | tail -1 | cut -c1-140)"
}

train() { # name, binary, args...
    local name=$1 bin=$2; shift 2
    timeout 900 "$BUILD/bin/$bin" -m "$MODEL_F32" -f "$TEXT" -c 128 -b 128 -ub 128 "$@" > $OUT/train_$name.log 2>&1
    local rc=$?
    local err=$(grep -E 'ASSERT|unsupported|not implemented|error' $OUT/train_$name.log | grep -v fit_params | head -1 | cut -c1-160)
    printf "%-22s rc=%-3s %s\n" "$name" "$rc" "$err"
    tr '\r' '\n' < $OUT/train_$name.log | grep 'train:' | grep -E 'data=0000(005|010|020)/' | sed 's/.*data=/  data=/' | cut -c1-90
}

echo "== 1. inference, FA off vs on, single vs TP"
ppl ref_fa_off       $REF -fa off
ppl tp_fa_off        $TP  -fa off
ppl tp_fa_on         $TP  -fa on

echo "== 2. LoRA training, FA off, single vs TP (loss should match to ~1e-4)"
train lora_ref llama-finetune-lora $REF -fa off --num-epochs 1 -s 1 --output-adapter $OUT/lora_ref.gguf
train lora_tp  llama-finetune-lora $TP  -fa off --num-epochs 1 -s 1 --output-adapter $OUT/lora_tp.gguf

echo "== 3. LoRA inference with the TP-trained adapter"
ppl lora_ref_adapter  $REF -fa off --lora $OUT/lora_ref.gguf
ppl lora_tp_adapter   $TP  -fa off --lora $OUT/lora_tp.gguf
ppl lora_tp_fa_on     $TP  -fa on  --lora $OUT/lora_tp.gguf

echo "== 4. full finetune, TP (F32 KV cache required over RPC)"
train full_ref llama-finetune $REF -ctk f32 -ctv f32
train full_tp  llama-finetune $TP  -ctk f32 -ctv f32

echo "logs in $OUT/"
