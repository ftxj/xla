#include "torch_xla/csrc/ops/leaky_relu_backward.h"

#include "torch_xla/csrc/elementwise.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {
namespace ir {
namespace ops {

LeakyReluBackward::LeakyReluBackward(const XlaValue& grad_output,
                                     const XlaValue& input,
                                     double negative_slope)
    : XlaNode(torch::lazy::OpKind(at::aten::leaky_relu_backward),
              {grad_output, input}, input.xla_shape(),
              /*num_outputs=*/1, torch::lazy::MHash(negative_slope)),
      negative_slope_(negative_slope) {}

torch::lazy::NodePtr LeakyReluBackward::Clone(OpList operands) const {
  return ir::MakeNode<LeakyReluBackward>(operands.at(0), operands.at(1),
                                         negative_slope_);
}

XlaOpVector LeakyReluBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp grad_output = loctx->GetOutputOp(operand(0));
  xla::XlaOp input = loctx->GetOutputOp(operand(1));
  xla::XlaOp output =
      BuildLeakyReluBackward(grad_output, input, negative_slope_);
  return ReturnOp(output, loctx);
}

std::string LeakyReluBackward::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", negative_slope=" << negative_slope_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
