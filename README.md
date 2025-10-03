# Project Comet bootstrap

This is a minimal working scaffold with CLI, tests, and CI that match the expectations you've shown.

## Dev quickstart
```bash
# optional: conda-forge compiled deps for healpy/namaster
micromamba create -n comet -c conda-forge python=3.11 healpy namaster pip -y
micromamba run -n comet python -m pip install -U pip
micromamba run -n comet pip install -e ".[dev]"

# verify
micromamba run -n comet comet demo
micromamba run -n comet pytest -q
```
