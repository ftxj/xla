#pragma once

#include "absl/types/span.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class UpdateSlice : public XlaNode {
 public:
  UpdateSlice(const XlaValue& input, const XlaValue& source,
              absl::Span<const int64_t> base_indices);

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& base_indices() const { return base_indices_; }

 private:
  std::vector<int64_t> base_indices_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
