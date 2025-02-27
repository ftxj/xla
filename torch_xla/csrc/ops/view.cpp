#include "torch_xla/csrc/ops/view.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const XlaValue& input,
                           absl::Span<const int64_t> output_sizes) {
  const xla::Shape& input_shape = input.xla_shape();
  auto info = XlaHelpers::GetDynamicReshapeInfo(input_shape, output_sizes);
  if (info) {
    return std::move(info->output_shape);
  }
  const auto complete_output_sizes =
      GetCompleteShape(output_sizes, input_shape.dimensions());
  return xla::ShapeUtil::MakeShape(input_shape.element_type(),
                                   complete_output_sizes);
}

}  // namespace

View::View(const XlaValue& input, std::vector<int64_t> output_size)
    : XlaNode(torch::lazy::OpKind(at::aten::view), {input},
              NodeOutputShape(input, output_size),
              /*num_outputs=*/1, torch::lazy::MHash(output_size)),
      output_size_(std::move(output_size)) {}

XlaOpVector View::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildView(input, output_size_);
  return ReturnOp(output, loctx);
}

std::string View::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", output_size=("
     << absl::StrJoin(output_size_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
