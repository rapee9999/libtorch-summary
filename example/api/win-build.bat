echo "Build project..." & \

@REM NOTE edit TORCH_DIR to libtorch's cmake directory path
set TORCH_DIR=D:\usr\rapee\projects\libtorch-win-shared-with-deps-debug-2.0.1+cu117\libtorch\share\cmake & \
echo "TORCH_DIR = %TORCH_DIR%" & \

@REM NOTE edit LIBTORCHSUM_DIR to libtorch-summary's cmake folder
set LIBTORCHSUM_DIR=D:\usr\rapee\projects\libtorch-summary\cmake & \
echo "LIBTORCHSUM_DIR = %LIBTORCHSUM_DIR%" & \

set PROJECT_DIR="example\api\.build" & \
mkdir %PROJECT_DIR% & \
cd %PROJECT_DIR% & \
echo "Change dir to %PROJECT_DIR%." & \

cmake ..
echo "Build project... done. Please compile."

@REM NOTE
@REM When open example/api/.build/example.sln in Visual Studio in order to compile the project, 
@REM set example project in Solution Explore to startup project.
@REM After compile, copy {libtorch directory}/lib/*.dll and lib/libtorch-summary.dll to example/api/.build/{Release or Debug}/.
@REM Run the compiled object exe in example/api/.build/{Release or Debug}/example.exe,
@REM see example/README.md in Run Example section.

@REM TODO
@REM linker error: external symbol to libtorch-summary.lib