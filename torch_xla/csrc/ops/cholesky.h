#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Cholesky : public XlaNode {
 public:
  Cholesky(const XlaValue& input, bool lower);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  bool lower() const { return lower_; }

 private:
  bool lower_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
