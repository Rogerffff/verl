#!/usr/bin/env bash
set -euo pipefail

ROOTS=""
CLAIMS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --roots)
      ROOTS="$2"
      shift 2
      ;;
    --claims)
      CLAIMS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$ROOTS" || -z "$CLAIMS" ]]; then
  echo "Usage: $0 --roots <path1,path2,...> --claims <claims_file>" >&2
  exit 2
fi

if [[ ! -f "$CLAIMS" ]]; then
  echo "Claims file not found: $CLAIMS" >&2
  exit 2
fi

IFS=',' read -r -a roots_array <<< "$ROOTS"
for r in "${roots_array[@]}"; do
  if [[ ! -d "$r" ]]; then
    echo "Root directory not found: $r" >&2
    exit 2
  fi
done

tmp_hits="/tmp/verify_claims_hits.txt"

echo "== Claim verification =="
echo "Roots: $ROOTS"
echo "Claims: $CLAIMS"
echo

while IFS= read -r claim; do
  [[ -z "$claim" ]] && continue
  [[ "${claim:0:1}" == "#" ]] && continue

  echo "[CLAIM] $claim"
  found_any=0

  for root in "${roots_array[@]}"; do
    echo "  Root: $root"

    if command -v rg >/dev/null 2>&1; then
      if rg -n --fixed-strings -- "$claim" "$root" > "$tmp_hits"; then
        found_any=1
        echo "    FOUND"
        sed -n '1,3p' "$tmp_hits" | sed 's/^/    - /'
        count=$(wc -l < "$tmp_hits" | tr -d ' ')
        if [[ "$count" -gt 3 ]]; then
          echo "    - ... ($count matches total)"
        fi
      else
        echo "    NOT FOUND"
      fi
    else
      if grep -R -n --fixed-strings -- "$claim" "$root" > "$tmp_hits" 2>/dev/null; then
        found_any=1
        echo "    FOUND"
        sed -n '1,3p' "$tmp_hits" | sed 's/^/    - /'
        count=$(wc -l < "$tmp_hits" | tr -d ' ')
        if [[ "$count" -gt 3 ]]; then
          echo "    - ... ($count matches total)"
        fi
      else
        echo "    NOT FOUND"
      fi
    fi
  done

  if [[ "$found_any" -eq 0 ]]; then
    echo "  => UNVERIFIED"
  fi

  echo
done < "$CLAIMS"
