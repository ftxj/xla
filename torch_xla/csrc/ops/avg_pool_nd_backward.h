#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class AvgPoolNdBackward : public XlaNode {
 public:
  AvgPoolNdBackward(const XlaValue& grad_output, const XlaValue& input,
                    int64_t spatial_dim_count, std::vector<int64_t> kernel_size,
                    std::vector<int64_t> stride, std::vector<int64_t> padding,
                    bool ceil_mode, bool count_include_pad);

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  int64_t spatial_dim_count() const { return spatial_dim_count_; }

  const std::vector<int64_t>& kernel_size() const { return kernel_size_; }

  const std::vector<int64_t>& stride() const { return stride_; }

  const std::vector<int64_t>& padding() const { return padding_; }

  bool ceil_mode() const { return ceil_mode_; }

  bool count_include_pad() const { return count_include_pad_; }

 private:
  int64_t spatial_dim_count_;
  // The parameters of the pooling.
  std::vector<int64_t> kernel_size_;
  std::vector<int64_t> stride_;
  std::vector<int64_t> padding_;
  bool ceil_mode_;
  // Whether the counts used to compute the average should include the added
  // padding.
  bool count_include_pad_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
