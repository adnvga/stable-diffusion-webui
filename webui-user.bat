@echo off

set PYTHON=C:\ai\python310\python
set GIT=C:\ai\git\bin\git

set COMMANDLINE_ARGS=--models-dir "C:\ai\models" --xformers --no-half-vae --listen
set COMMANDLINE_ARGS=--models-dir "C:\ai\models" --xformers --no-half-vae

set PIP_BUILD_CONSTRAINT=%~dp0build-constraints.txt
set SD_WEBUI_MEMORY_DEBUG=0
set SD_WEBUI_VAE_TILING=1

call webui.bat
