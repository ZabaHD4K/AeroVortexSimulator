@echo off
set "MSVC_DIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207"
set "WINSDK_DIR=C:\Program Files (x86)\Windows Kits\10"
set "WINSDK_VER=10.0.26100.0"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"

set "PATH=%MSVC_DIR%\bin\HostX64\x64;%WINSDK_DIR%\bin\%WINSDK_VER%\x64;%CUDA_PATH%\bin;C:\Program Files\CMake\bin;C:\Users\aleja\AppData\Local\Microsoft\WinGet\Packages\Ninja-build.Ninja_Microsoft.Winget.Source_8wekyb3d8bbwe;%PATH%"
set "INCLUDE=%MSVC_DIR%\include;%WINSDK_DIR%\Include\%WINSDK_VER%\ucrt;%WINSDK_DIR%\Include\%WINSDK_VER%\shared;%WINSDK_DIR%\Include\%WINSDK_VER%\winrt;%WINSDK_DIR%\Include\%WINSDK_VER%\um;%CUDA_PATH%\include"
set "LIB=%MSVC_DIR%\lib\x64;%WINSDK_DIR%\Lib\%WINSDK_VER%\ucrt\x64;%WINSDK_DIR%\Lib\%WINSDK_VER%\um\x64;%CUDA_PATH%\lib\x64"

if not exist build mkdir build
cd build

if not exist build.ninja (
    cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl -DCMAKE_RC_COMPILER=rc -DCMAKE_MT=mt
    if %errorlevel% neq 0 exit /b 1
)

cmake --build .
exit /b %errorlevel%
