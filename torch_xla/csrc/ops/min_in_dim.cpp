#include "torch_xla/csrc/ops/min_in_dim.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/reduction.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const XlaValue& input, int64_t dim, bool keepdim) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    xla::XlaOp values = BuildMinInDim(operands[0], dim, keepdim);
    xla::XlaOp indices = BuildArgMin(operands[0], dim, keepdim);
    return xla::Tuple(values.builder(), {values, indices});
  };
  return InferOutputShape({input.xla_shape()}, lower_for_shape_fn);
}

}  // namespace

MinInDim::MinInDim(const XlaValue& input, int64_t dim, bool keepdim)
    : XlaNode(torch::lazy::OpKind(at::aten::min), {input},
              [&]() { return NodeOutputShape(input, dim, keepdim); },
              /*num_outputs=*/2, torch::lazy::MHash(dim, keepdim)),
      dim_(dim),
      keepdim_(keepdim) {}

torch::lazy::NodePtr MinInDim::Clone(OpList operands) const {
  return ir::MakeNode<MinInDim>(operands.at(0), dim_, keepdim_);
}

XlaOpVector MinInDim::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp values = BuildMinInDim(input, dim_, keepdim_);
  xla::XlaOp indices = BuildArgMin(input, dim_, keepdim_);
  return ReturnOps({values, indices}, loctx);
}

std::string MinInDim::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", dim=" << dim_ << ", keepdim=" << keepdim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
