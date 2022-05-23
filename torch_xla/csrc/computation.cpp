#include "torch_xla/csrc/computation.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include <iostream>
namespace torch_xla {

Computation::Computation(std::string name, xla::XlaComputation computation)
    : name_(std::move(name)), computation_(std::move(computation)) {
  std::cout << "[FTXJ LOG] Computation Construction" << std::endl;
  std::cout << "[FTXJ MSG] name=" << name << std::endl;
  std::cout << "[FTXJ LOG] Computation Construction call ConsumeValue to get program_shape" << std::endl;
  program_shape_ = ConsumeValue(computation_.GetProgramShape());
  std::cout << "[FTXJ LOG] Computation Construction call ConsumeValue End" << std::endl;
  hash_ = torch::lazy::MHash(name_, computation_.proto().SerializeAsString());
}

}  // namespace torch_xla
