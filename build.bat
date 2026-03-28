@echo off
setlocal enabledelayedexpansion
cd /d C:\Users\drew\cursdar

echo === CURSDAR Build ===
echo.

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64 >nul 2>&1
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin;%PATH%"

if not exist build mkdir build
cd build

cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release 2>&1
if errorlevel 1 (
    echo ERROR: CMake configuration failed
    exit /b 1
)

cmake --build . -j 2>&1
if errorlevel 1 (
    echo ERROR: Build failed
    exit /b 1
)

echo.
echo === Build successful ===
