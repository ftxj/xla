#include "torch_xla/csrc/ops/upsample_nearest2d.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/resize_ops.h"

namespace torch_xla {
namespace ir {
namespace ops {

UpsampleNearest::UpsampleNearest(const XlaValue& input,
                                 std::vector<int64_t> output_size)
    : XlaNode(torch::lazy::OpKind(at::aten::upsample_nearest2d), {input},
              [&]() {
                return resize::GetForwardOutputShape2d(input.xla_shape(),
                                                       output_size);
              },
              /*num_outputs=*/1, torch::lazy::MHash(output_size)),
      output_size_(std::move(output_size)) {}

torch::lazy::NodePtr UpsampleNearest::Clone(OpList operands) const {
  return ir::MakeNode<UpsampleNearest>(operands.at(0), output_size_);
}

XlaOpVector UpsampleNearest::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output =
      resize::LowerForward2d("ResizeNearest", input, xla_shape(),
                             /*align_corners=*/false,
                             /*half_pixel_centers=*/false);
  return ReturnOp(output, loctx);
}

std::string UpsampleNearest::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", output_size=("
     << absl::StrJoin(output_size_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
