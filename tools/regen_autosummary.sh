#!/usr/bin/env bash
# Regenerate docs/api/_autosummary/ stubs from the current source tree.
#
# Run from repo root in cstm_scvi_env (or any env where the real classes
# import — NOT a mocks-only RTD-style env, or the stubs will list
# MagicMock dunders instead of real methods).
#
# The stubs are committed to git and read at build time when
# `autosummary_generate = False` in docs/conf.py — which is the default
# RTD config. Without this script you'd have to flip the flag, build,
# flip it back, and commit by hand. Do that here once and the docs pick
# up new public classes / functions automatically.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# Wipe old stubs so removed names don't linger.
rm -rf docs/api/_autosummary

# Flip on stub generation just for this build.
sed -i.bak 's/^autosummary_generate = False.*/autosummary_generate = True/' docs/conf.py

# Generate.
sphinx-build -b html docs docs/_build/html

# Flip back.
sed -i 's/^autosummary_generate = True.*/autosummary_generate = False  # stubs are pre-generated and committed under docs\/api\/_autosummary\//' docs/conf.py
rm -f docs/conf.py.bak

N=$(ls docs/api/_autosummary | wc -l)
echo "regenerated $N stubs under docs/api/_autosummary/"
echo "review with: git diff --stat docs/api/_autosummary/"
echo "commit with: git add docs/api/_autosummary/ docs/conf.py && git commit -m 'docs: regen autosummary stubs'"
