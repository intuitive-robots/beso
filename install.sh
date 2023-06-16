#!/bin/bash

############ GENERAL ENV SETUP ############
echo New Environment Name:
read envname

echo Creating new conda environment $envname 
conda create -n $envname python=3.8 -y -q

eval "$(conda shell.bash hook)"
conda activate $envname

echo
echo Activating $envname
if [[ "$CONDA_DEFAULT_ENV" != "$envname" ]]
then
    echo Failed to activate conda environment.
    exit 1
fi


############ PYTHON ############
echo Install mamba
conda install mamba -c conda-forge -y -q


############ REQUIRED DEPENDENCIES (PYBULLET) ############
echo Installing dependencies...

mamba install -c conda-forge hydra-core -y -q
# Mujoco System Dependencies
mamba install -c conda-forge glew patchelf -y -q
mamba install conda-build -y -q
# Set Conda Env Variables
conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
conda env config vars set LD_PRELOAD=$LD_PRELOAD:$CONDA_PREFIX/lib/libGLEW.so
# Activate Mujoco Py Env Variables
conda activate $envname

# Install MujocoPy
pip install mujoco-py

# Install other PIP Dependencies
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
pip install setuptools==65.5.0
pip install wheel==0.38.4
pip install dm-control==0.0.403778684
pip install gym==0.21.0
pip install termcolor
pip install wandb
pip install tikzplotlib
pip install einops
pip install torchdiffeq
pip install gin-config
pip install pybullet
pip install -U scikit-learn
pip install torchsde

echo Clone Relay policy learning
git clone https://github.com/google-research/relay-policy-learning

echo Done installing all necessary packages. Please follow the next steps mentioned on the readme

pip install -e .

echo
echo
echo Successfully installed.
echo
echo To activate your environment call:
echo conda activate $envname
exit 0 
