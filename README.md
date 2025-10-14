# czi-minimal
Minimal Example for Pretraining PCad1 with composer on CZI repo. Meant to test deployment.

Easiest way to setup is to:
1. Clone this repo
2. Clone Eric's branch of caducues (fork of main public repo)
3. Update the python path for the env being used (assume conda for now, perhaps switch to ```uv``` later.)

*Assumes we already have an existing env on the pvc we can activate*, env set-up not included here. 
```bash
mkdir -p repos
cd repos
git clone eric-czech:caduceus.git
cd caduceus
git checkout eac-caduceus-act-ckpt
cd ..
PATH_TO_CAD=${realpath caduceus}
PATH_TO_CONDA_ENV="/share/kuleshov/emm392/conda/conda_envs"
ENV_NAME="llm_lib"

echo ${PATH_TO_CAD} >> ${PATH_TO_CONDA_ENV}/${ENV_NAME}/lib/python3.12/site-packages/${$ENV_NAME}.pth
```
