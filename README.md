# czi-minimal
Minimal Example for Pretraining DNA LMs with composer on CZI repo. Meant to test deployment and serve as a launch point for experiments, active dev occurs in another repo that eventually pushes into this dir.

We use uv to manage the env, and we have two submodules (caduceus and hnet). 
Easiest way to setup is to:
1. Clone this repo
2. Run the following commands
```bash
# init the submodule (personal forks to fix pyproject.toml's)
git submodule update --init --recursive
# update the env
uv sync
# run an example experiment
NUM_GPUS=1
uv run composer -n ${NUM_GPUS} main.py experiment=debug_flops
```
Should work fine from scratch, but `flash-attn` might give some problems due to needing no-build-isolation. Exclude from the first sync and running a second sync after __should__ fix things.


## Code of Conduct

This project adheres to the Contributor Covenant [code of conduct](https://github.com/chanzuckerberg/.github/blob/master/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [opensource@chanzuckerberg.com](mailto:opensource@chanzuckerberg.com).

## Reporting Security Issues

If you believe you have found a security issue, please responsibly disclose by contacting us at [security@chanzuckerberg.com](mailto:security@chanzuckerberg.com).
