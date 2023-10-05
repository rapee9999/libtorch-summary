echo "Build project..." && \

# NOTE edit TORCH_DIR to libtorch's cmake directory path
export TORCH_DIR="/opt/libtorch/share/cmake" && \
echo "TORCH_DIR = ${TORCH_DIR}" && \

export PROJECT_DIR=.build && \
mkdir ${PROJECT_DIR} || \
cd ${PROJECT_DIR} && \
echo "Change dir to ${PROJECT_DIR}." && \

cmake .. && \
echo "Build project... done. Please compile."