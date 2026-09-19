@echo off

set PYTHON=C:\ai\python310\python
set GIT=C:\ai\git\bin\git

set COMMANDLINE_ARGS=--models-dir "C:\ai\models" --xformers --no-half-vae --listen
set COMMANDLINE_ARGS=--models-dir "C:\ai\models" --xformers --no-half-vae

set PIP_BUILD_CONSTRAINT=%~dp0build-constraints.txt

call webui.bat
