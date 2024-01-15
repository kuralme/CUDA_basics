# CUDA Basics
Contains scripts of basic functions that can be fastened with spatial programming, specifically with CUDA kernels. There are various naive and optimised version implementations to test their execution time from launching the kernel to obtaining the result data in the host.

Codes are built using CMake and GNU make on WSL Ubuntu.

## Requirements
An NVIDIA CUDA Capable GPU
Matching CUDA Driver and CUDA Toolkit
CMake
GNU Make

## Installation
Clone the repo on your workspace and run the script:
```
bash makebuild.sh
```
This will create executables specified in 'CMakeList.txt' file

If using VSCode add following to folder '.vscode' as 'c_cpp_properties.json' file to let Intellisense find nvcc
```
{
    "configurations": [
        {
            "name": "Linux",
            "includePath": [
                "${workspaceFolder}/**"
            ],
            "compilerPath": "/usr/local/cuda/bin/nvcc",
            "defines": [],
            "cStandard": "gnu17",
            "cppStandard": "gnu++14",
            "intelliSenseMode": "linux-gcc-x64"
        }
    ],
    "version": 4
}
```
