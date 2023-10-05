# Example
This folder demonstrates how to use libtorch-summary.

**Outline**
1. [Build Examples](#build-examples)
1. [Run Examples](#run-examples)

## Build Examples
1. Clone this git repo in your preferred directory by running below command.<br>
    `git clone https://github.com/rapee9999/libtorch-summary.git`
1. Change directory to the repo by running below command.<br>
    `cd libtorch-summary`
1. Make sure you have C++ compiler (e.g. `gcc`) and `libtorch`, or run prepared docker image as below.<br>
    `docker-compose up -d`<br>
    `docker exec -it libtorch-summary-dev bash`
1. Change directory to example folder.<br>
    - For application using libtorch-summary shared library,<br>
    `cd example/api`
    - For application built together with libtorch-summary source code,<br>
    `cd example/bundle`
1. Build the example application.<br>
    - For Linux, **please read note** in `linux-build.sh` file before run following commands.<br>
    `./linux-build.sh`<br>
    `cd .build`<br>
    `make`
    - For Windows, **please read note** in `win-build.bat` file before run following command.<br>
    `./win-build`<br>
    After project created, open and build `.build/libtorch-summary.sln` in Visual Studio.
1. If build successfully, you will find executable file in `.build` folder. `example` executable file, download model.<br>
    - For Linux, `example`.
    - For Windows, `example.exe`.
1. If you run the prepared docker image, you can terminate the container by running below command.<br>
`docker-compose down`


## Run Examples
1. Change directory to this repo root folder.
1. Before run the application, prepare PyTorch model as following.
    1. Install Python 3.10.
    1. Install packages by running `pip install -r example/model-store/requirements.txt`.
    1. Run `python example/model-store/save_dcgan.py`
    1. And you will find `example/model-store/jit_script_dcgan.pt`.
1. Run example applications by one of below commands.    
    - example/api Linux version:<br>
    `./example/api/.build/example example/model-store/jit_script_dcgan.pt 1 100 1 1 24`
    - example/api Windows version<br>
    `./example/api/.build/Debug/example example/model-store/jit_script_dcgan.pt 1 100 1 1 24`
    - example/bundle Linux version<br>
    `./example/bundle/.build/example example/model-store/jit_script_dcgan.pt 1 100 1 1 24`
    - example/bundle Windows version<br>
    `./example/bundle/.build/Debug/example example/model-store/jit_script_dcgan.pt 1 100 1 1 24`

    > *Arguments:*<br>
    First: Number of input data<br>
    Second: Number of channel<br>
    Third: Height<br>
    Fourth: Width<br>
    Fifth: Table column width
