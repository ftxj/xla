#include "torch_xla/csrc/ops/expand.h"

#include "absl/strings/str_join.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

#include <iostream>

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const XlaValue& input,
                           const std::vector<int64_t>& size) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildExpand(operands[0], size);
  };

  std::cout << "[FTXJ LOG] NodeOutputShape" << std::endl;
  std::cout << "[FTXJ LOG] NodeOutputShape call InferOutputShape" << std::endl;
  
  auto tmp = InferOutputShape({input.xla_shape()}, lower_for_shape_fn);
  
  std::cout << "[FTXJ LOG] NodeOutputShape End" << std::endl;
  return tmp;
}

}  // namespace

Expand::Expand(const XlaValue& input, std::vector<int64_t> size)
    : XlaNode(torch::lazy::OpKind(at::aten::expand), {input},
              [&]() { return NodeOutputShape(input, size); },
              /*num_outputs=*/1, torch::lazy::MHash(size)),
      size_(std::move(size)) {
        std::cout << "Construct Expand Node End" << std::endl;
      }

torch::lazy::NodePtr Expand::Clone(OpList operands) const {
  return ir::MakeNode<Expand>(operands.at(0), size_);
}

XlaOpVector Expand::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildExpand(input, size_), loctx);
}

std::string Expand::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", size=(" << absl::StrJoin(size_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
