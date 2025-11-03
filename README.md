# czi-minimal
Minimal Example for Pretraining PCad1 with composer on CZI repo. Meant to test deployment.

Easiest way to setup is to:
1. Clone this repo
2. Clone Caduceus
3. Update the python path for the env being used (assume conda for now, switch to ```uv``` later pending what our compute env looks like)

*Assumes we already have an existing env on the pvc we can activate*, env set-up not included here. 
```bash
mkdir -p repos
cd repos
git clone https://github.com/kuleshov-group/caduceus.git
# Add to python path
PATH_TO_CAD=${realpath caduceus}
PATH_TO_CONDA_ENV="/path/to/conda_envs"
ENV_NAME="env_name"

echo ${PATH_TO_CAD} >> ${PATH_TO_CONDA_ENV}/${ENV_NAME}/lib/python3.12/site-packages/${$ENV_NAME}.pth
```
