#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="${SCRIPT_DIR}/.."
LOGDIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOGDIR"

seed=""
seeds=""
data=""
method=""
with_survival_cli=""
other_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed=*)
      seed="${1#*=}"; shift ;;
    --seed)
      seed="$2"; shift 2 ;;

    --seeds=*)
      seeds="${1#*=}"; shift ;;
    --seeds)
      seeds="$2"; shift 2 ;;

    --data=*|--dataset=*)
      data="${1#*=}"; shift ;;
    --data|--dataset)
      data="$2"; shift 2 ;;

    --method=*|--estimator=*)
      method="${1#*=}"; shift ;;
    --method|--estimator)
      method="$2"; shift 2 ;;

    --with_survival=*)
      with_survival_cli="${1#*=}"; shift ;;
    --with_survival)
      with_survival_cli="$2"; shift 2 ;;

    *)
      other_args+=("$1"); shift ;;
  esac
done

data="${data%\"}"; data="${data#\"}"
data="${data%\'}"; data="${data#\'}"

if [[ -z "$data" && -n "${DATA:-}" ]]; then
  data="$DATA"
fi
if [[ -z "$data" ]]; then
  echo "[run.sh] ERROR: --data/--dataset is required." >&2
  exit 1
fi

if [[ -z "$method" && -n "${METHOD:-}" ]]; then
  method="$METHOD"
fi
if [[ -z "$method" ]]; then
  echo "[run.sh] ERROR: --method/--estimator is required." >&2
  exit 1
fi

# -----------------------
# -----------------------
if [[ -n "$seeds" ]]; then
  IFS=',' read -r -a seed_list <<< "$seeds"
elif [[ -n "$seed" ]]; then
  seed_list=("$seed")
elif [[ -n "${SEED:-}" ]]; then
  seed_list=("$SEED")
else
  seed_list=("1")
fi

# -----------------------
# -----------------------
train_able="${TRAIN_ABLE:-1}"
batch_size="${BATCH_SIZE:-64}"
n_head="${N_HEAD:-3}"
n_layers="${N_LAYERS:-2}"
d_model="${D_MODEL:-36}"
d_inner_hid="${D_INNER_HID:-${D_INNER:-8}}"
d_k="${D_K:-16}"
d_v="${D_V:-16}"
dropout="${DROPOUT:-0.1}"
lr="${LR:-1e-3}"
epoch="${EPOCH:-100}"
CE_coef="${CE_COEF:-10.0}"
normalize_time="${NORMALIZE_TIME:-1}"
alpha_survival="${ALPHA_SURVIVAL:-10.0}"
num_grid="${NUM_GRID:-10}"
noise_var="${NOISE_VAR:-0.5}"
num_noise="${NUM_NOISE:-50}"
noise_type="${NOISE_TYPE:-lognormal}"
normalize_scale="${NORMALIZE_SCALE:-50.0}"
alpha_neg="${ALPHA_NEG:-1.0}"
with_survival_env="${WITH_SURVIVAL:-1}"

if [[ -n "$with_survival_cli" ]]; then
  with_survival="$with_survival_cli"
else
  with_survival="$with_survival_env"
fi

log_name="${LOG:-log.txt}"
model="${MODEL:-sahp}"
load_model="${LOAD_MODEL:-0}"

has_stdbuf=0
if command -v stdbuf >/dev/null 2>&1; then
  has_stdbuf=1
fi

clean_data="${data//\//_}"

for s in "${seed_list[@]}"; do
  ts=$(date +%F_%H-%M-%S)
  log_file="${LOGDIR}/${clean_data}_${method}_seed${s}_${ts}.log"

  echo "[run.sh] data=${data} method=${method} seed=${s} with_survival=${with_survival}"
  echo "[run.sh] log => ${log_file}"

  if [[ $has_stdbuf -eq 1 ]]; then
    (
      cd "$PROJ_DIR"
      stdbuf -oL -eL python -u Main.py \
        -data "${data}" \
        -epoch "${epoch}" \
        -batch_size "${batch_size}" \
        -d_model "${d_model}" \
        -d_inner_hid "${d_inner_hid}" \
        -d_k "${d_k}" \
        -d_v "${d_v}" \
        -n_head "${n_head}" \
        -n_layers "${n_layers}" \
        -dropout "${dropout}" \
        -lr "${lr}" \
        -log "${log_name}" \
        -mode train \
        -method "${method}" \
        -model "${model}" \
        -train_able "${train_able}" \
        -load_model "${load_model}" \
        -num_grid "${num_grid}" \
        -CE_coef "${CE_coef}" \
        -noise_var "${noise_var}" \
        -num_noise "${num_noise}" \
        -seed "${s}" \
        -normalize_time "${normalize_time}" \
        -normalize_scale "${normalize_scale}" \
        -alpha_survival "${alpha_survival}" \
        -noise_type "${noise_type}" \
        -with_survival "${with_survival}" \
        -alpha_neg "${alpha_neg}" \
        "${other_args[@]}"
    ) |& tee "${log_file}"
  else
    (
      cd "$PROJ_DIR"
      python -u Main.py \
        -data "${data}" \
        -epoch "${epoch}" \
        -batch_size "${batch_size}" \
        -d_model "${d_model}" \
        -d_inner_hid "${d_inner_hid}" \
        -d_k "${d_k}" \
        -d_v "${d_v}" \
        -n_head "${n_head}" \
        -n_layers "${n_layers}" \
        -dropout "${dropout}" \
        -lr "${lr}" \
        -log "${log_name}" \
        -mode train \
        -method "${method}" \
        -model "${model}" \
        -train_able "${train_able}" \
        -load_model "${load_model}" \
        -num_grid "${num_grid}" \
        -CE_coef "${CE_coef}" \
        -noise_var "${noise_var}" \
        -num_noise "${num_noise}" \
        -seed "${s}" \
        -normalize_time "${normalize_time}" \
        -normalize_scale "${normalize_scale}" \
        -alpha_survival "${alpha_survival}" \
        -noise_type "${noise_type}" \
        -with_survival "${with_survival}" \
        -alpha_neg "${alpha_neg}" \
        "${other_args[@]}"
    ) |& tee "${log_file}"
  fi

done
