#include "torch_xla/csrc/ops/mse_loss.h"

#include <ATen/core/Reduction.h>

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch/csrc/lazy/core/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const XlaValue& input, const XlaValue& target,
                           ReductionMode reduction) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildMseLoss(operands[0], operands[1], reduction);
  };
  return InferOutputShape({input.xla_shape(), target.xla_shape()},
                          lower_for_shape_fn);
}

}  // namespace

MseLoss::MseLoss(const XlaValue& input, const XlaValue& target,
                 ReductionMode reduction)
    : XlaNode(torch::lazy::OpKind(at::aten::mse_loss), {input, target},
              [&]() { return NodeOutputShape(input, target, reduction); },
              /*num_outputs=*/1,
              torch::lazy::MHash(torch::lazy::GetEnumValue(reduction))),
      reduction_(reduction) {}

torch::lazy::NodePtr MseLoss::Clone(OpList operands) const {
  return ir::MakeNode<MseLoss>(operands.at(0), operands.at(1), reduction_);
}

XlaOpVector MseLoss::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp target = loctx->GetOutputOp(operand(1));
  return ReturnOp(BuildMseLoss(input, target, reduction_), loctx);
}

std::string MseLoss::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString()
     << ", reduction=" << torch::lazy::GetEnumValue(reduction_);
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
