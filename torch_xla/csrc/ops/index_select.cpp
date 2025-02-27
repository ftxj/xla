#include "torch_xla/csrc/ops/index_select.h"

#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const XlaValue& input, const XlaValue& index,
                           int64_t dim) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return xla::TorchIndexSelect(operands[0], operands[1], dim);
  };
  return InferOutputShape({input.xla_shape(), index.xla_shape()},
                          lower_for_shape_fn);
}

}  // namespace

IndexSelect::IndexSelect(const XlaValue& input, int64_t dim,
                         const XlaValue& index)
    : XlaNode(torch::lazy::OpKind(at::aten::index_select), {input, index},
              [&]() { return NodeOutputShape(input, index, dim); },
              /*num_outputs=*/1, torch::lazy::MHash(dim)),
      dim_(dim) {}

torch::lazy::NodePtr IndexSelect::Clone(OpList operands) const {
  return ir::MakeNode<IndexSelect>(operands.at(0), dim_, operands.at(1));
}

XlaOpVector IndexSelect::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp index = loctx->GetOutputOp(operand(1));
  return ReturnOp(xla::TorchIndexSelect(input, index, dim_), loctx);
}

std::string IndexSelect::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
