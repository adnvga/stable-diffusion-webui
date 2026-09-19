@echo off
setlocal

echo Buscando directorios __pycache__ desde:
echo %~dp0
echo.

for /d /r "%~dp0" %%D in (__pycache__) do (
    if exist "%%D" (
        echo Eliminando: "%%D"
        rd /s /q "%%D"
    )
)

echo.
echo Limpieza terminada.
pause