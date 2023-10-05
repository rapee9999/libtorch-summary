//
// Mimick Python's torchsummary
//

#include <iostream>
#include <vector>
#include <torch/torch.h>
#include <torch/script.h>
#include <inttypes.h>
#include <memory>
#include <string.h>
#include <any>


namespace torchsummary {

    /*
     * Print summary of the given model.
     *
     * @param module Shared pointer to a JIT module.
     * @param inputShape Dimension of input tensor.
     * @param name Model name
     * @param cellWidth Cell width in number of charactors
     */
    void summary(
        std::shared_ptr<torch::jit::Module> module, 
        std::vector<int64_t> inputShape = {}, 
        std::string name = "Total", 
        int cellWidth = 16 );

}