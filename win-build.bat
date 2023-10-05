echo "Build project..." & \

@REM NOTE edit TORCH_DIR to libtorch's cmake directory path
set TORCH_DIR=D:\usr\rapee\projects\libtorch-win-shared-with-deps-debug-2.0.1+cu117\libtorch\share\cmake & \
echo "TORCH_DIR = %TORCH_DIR%" & \

set PROJECT_DIR=".build" & \
mkdir %PROJECT_DIR% & \
cd %PROJECT_DIR% & \
echo "Change dir to %PROJECT_DIR%." & \

cmake ..
echo "Build project... done. Please compile."
@REM NOTE
@REM When open .build/libtorch-summary.sln in Visual Studio in order to compile the project, 
@REM set example project in Solution Explore to startup project.
@REM After compile, copy bin/{Release or Debug}/libtorch-summary.dll and
@REM /{Release or Debug}/libtorch-summary.lib to lib folder.