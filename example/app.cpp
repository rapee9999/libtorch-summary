//
// libtorch-summary usage
//

#include <iostream>
#include <torch/torch.h>
#include <ctime>
#include <memory>
#include "libtorch-summary.h"
#include <string>
#include <stdlib.h>

using namespace torch;
using namespace torchsummary;


int main(int argc, char * argv[]) {
    // validate arguments
    if (argc <= 1) {
        std::cout << "Please enter model path and input dimension" << std::endl;
        return 1;
    }

    // read model
    torch::jit::Module scriptModel;
    try {
        scriptModel = torch::jit::load(argv[1]);
    }
    catch (c10::Error e) {
        std::cout << e.msg() << std::endl;
        return 1;
    }

    // take file name as model name
    std::string strModelName = std::string(argv[1]);
    strModelName = strModelName.substr(strModelName.find_last_of("/\\") + 1); // handle only linux path style
    strModelName = strModelName.substr(0, strModelName.find_last_of("."));

    // get input size
    std::vector<int64_t> inputSize;
    if (argc >= 6) {
        inputSize = {strtol(argv[2], NULL, 10), strtol(argv[3], NULL, 10), strtol(argv[4], NULL, 10), strtol(argv[5], NULL, 10)};
    }
    else {
        inputSize = {};
    }

    // get summary table's cell width
    int cellWidth;
    if (argc >= 7) {
        cellWidth = strtol(argv[6], NULL, 10);
    }
    else {
        cellWidth = 16;
    }

    // init device
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Initialize device :: " << device << std::endl;
    
    // summary
    std::shared_ptr<torch::jit::Module> shPtrModule = std::make_shared<torch::jit::Module>(scriptModel);
    summary(shPtrModule, inputSize, strModelName, cellWidth);

    return 0;
}
