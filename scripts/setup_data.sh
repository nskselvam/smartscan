#!/usr/bin/env bash
set -euo pipefail

REPO_ID="alan-turing-institute/turing-synthetic-radar-dataset"
MODE="scan"
SPLIT="train"
INCLUDE_PATTERN=""
MIN_FREE_GB=10

usage() {
    cat <<'EOF'
Usage: scripts/setup_data.sh [options]

Prepare local SMARTSCAN dataset directories and optionally download an explicit
TSRD file pattern using the official Hugging Face CLI.

Options:
  --mode scan|stare        Dataset collection mode (default: scan)
  --split train|validation|test
                            Dataset split label for the local destination
  --include PATTERN        Hugging Face allow-pattern to download
  --min-free-gb NUMBER     Required available disk space (default: 10)
  -h, --help               Show this help

The script never prints HUGGINGFACE_TOKEN. TSRD is gated: first accept its
Hugging Face access conditions, then use an existing `hf auth login` session or
HUGGINGFACE_TOKEN when access requires authentication.

Pass --include only after verifying the repository file paths from the official
Hugging Face file browser.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --include) INCLUDE_PATTERN="$2"; shift 2 ;;
        --min-free-gb) MIN_FREE_GB="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$MODE" in scan|stare) ;; *) echo "--mode must be scan or stare" >&2; exit 2 ;; esac
case "$SPLIT" in train|validation|test) ;; *) echo "--split must be train, validation, or test" >&2; exit 2 ;; esac

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target_dir="$project_root/data/raw/tsrd_${MODE}_${SPLIT}"
mkdir -p "$target_dir" "$project_root/data/processed" "$project_root/data/cache"

if ! command -v cargo >/dev/null 2>&1; then
    echo "Rust/Cargo is required. Install Rust from https://rustup.rs/." >&2
    exit 1
fi

if [[ -x "$project_root/.venv/bin/hf" ]]; then
    hf_command=("$project_root/.venv/bin/hf" download "$REPO_ID" --repo-type dataset --local-dir "$target_dir")
elif command -v hf >/dev/null 2>&1; then
    hf_command=(hf download "$REPO_ID" --repo-type dataset --local-dir "$target_dir")
elif command -v huggingface-cli >/dev/null 2>&1; then
    hf_command=(huggingface-cli download "$REPO_ID" --repo-type dataset --local-dir "$target_dir")
else
    echo "Install the official Hugging Face CLI before downloading: pipx install huggingface_hub" >&2
    exit 1
fi

available_kb="$(df -Pk "$project_root" | awk 'NR == 2 { print $4 }')"
required_kb=$((MIN_FREE_GB * 1024 * 1024))
if (( available_kb < required_kb )); then
    echo "Insufficient disk space: need at least ${MIN_FREE_GB} GB free." >&2
    exit 1
fi

if [[ -z "$INCLUDE_PATTERN" ]]; then
    echo "Directories prepared at $target_dir"
    echo "No download started. Accept TSRD's gated-access conditions and verify file paths, then rerun with --include PATH_PATTERN."
    echo "Use cargo run -- data inspect --dataset tsrd_${MODE}_${SPLIT} after downloading."
    exit 0
fi

if [[ -n "${HUGGINGFACE_TOKEN:-}" ]]; then
    export HF_TOKEN="$HUGGINGFACE_TOKEN"
fi

"${hf_command[@]}" --include "$INCLUDE_PATTERN"
cargo run --release -- data inspect --dataset "tsrd_${MODE}_${SPLIT}"
du -sh "$target_dir"
