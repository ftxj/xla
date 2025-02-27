#pragma once

#include <vector>

#include "tensorflow/compiler/xla/types.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Std : public XlaNode {
 public:
  Std(const XlaValue& input, std::vector<int64_t> dimensions,
      bool keep_reduced_dimensions, int64_t correction);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const std::vector<int64_t>& dimensions() const { return dimensions_; }

  bool keep_reduced_dimensions() const { return keep_reduced_dimensions_; }

  int64_t correction() const { return correction_; }

 private:
  std::vector<int64_t> dimensions_;
  bool keep_reduced_dimensions_;
  int64_t correction_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
