#!/usr/bin/env bash
set -euo pipefail

ROOT=""
SYMBOLS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT="$2"
      shift 2
      ;;
    --symbols)
      SYMBOLS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$ROOT" || -z "$SYMBOLS" ]]; then
  echo "Usage: $0 --root <repo_root> --symbols <symbols_file>" >&2
  exit 2
fi

if [[ ! -d "$ROOT" ]]; then
  echo "Root directory not found: $ROOT" >&2
  exit 2
fi

if [[ ! -f "$SYMBOLS" ]]; then
  echo "Symbols file not found: $SYMBOLS" >&2
  exit 2
fi

echo "== Symbol verification =="
echo "Root: $ROOT"
echo "Symbols: $SYMBOLS"
echo

while IFS= read -r symbol; do
  [[ -z "$symbol" ]] && continue
  [[ "${symbol:0:1}" == "#" ]] && continue

  echo "[SYMBOL] $symbol"
  if command -v rg >/dev/null 2>&1; then
    if rg -n --fixed-strings -- "$symbol" "$ROOT" >/tmp/verify_symbols_hits.txt; then
      echo "  FOUND"
      sed -n '1,5p' /tmp/verify_symbols_hits.txt | sed 's/^/  - /'
      count=$(wc -l < /tmp/verify_symbols_hits.txt | tr -d ' ')
      if [[ "$count" -gt 5 ]]; then
        echo "  - ... ($count matches total)"
      fi
    else
      echo "  NOT FOUND"
    fi
  else
    if grep -R -n --fixed-strings -- "$symbol" "$ROOT" >/tmp/verify_symbols_hits.txt 2>/dev/null; then
      echo "  FOUND"
      sed -n '1,5p' /tmp/verify_symbols_hits.txt | sed 's/^/  - /'
      count=$(wc -l < /tmp/verify_symbols_hits.txt | tr -d ' ')
      if [[ "$count" -gt 5 ]]; then
        echo "  - ... ($count matches total)"
      fi
    else
      echo "  NOT FOUND"
    fi
  fi
  echo
done < "$SYMBOLS"
