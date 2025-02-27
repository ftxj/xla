#pragma once

#include "torch_xla/csrc/cross_replica_reduces.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class CollectivePermute : public XlaNode {
 public:
  CollectivePermute(
      const XlaValue& input, const XlaValue& token,
      std::vector<std::pair<int64_t, int64_t>> source_target_pairs);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs() const {
    return source_target_pairs_;
  }

 private:
  std::vector<std::pair<int64_t, int64_t>> source_target_pairs_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
