#include "torch_xla/csrc/ops/cumprod.h"

#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "torch/csrc/lazy/core/tensor_util.h"
#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/reduction.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {
namespace {

xla::XlaOp LowerCumProd(xla::XlaOp input, int64_t dim,
                        c10::optional<at::ScalarType> dtype) {
  xla::XlaOp casted_input = CastToScalarType(input, dtype);
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(casted_input);
  xla::XlaOp init =
      xla::One(casted_input.builder(), input_shape.element_type());
  xla::XlaComputation reducer =
      XlaHelpers::CreateMulComputation(input_shape.element_type());
  return BuildCumulativeComputation(casted_input, dim, reducer, init);
}

xla::Shape NodeOutputShape(const XlaValue& input,
                           c10::optional<at::ScalarType> dtype) {
  if (dtype) {
    return xla::ShapeUtil::ChangeElementType(
        input.xla_shape(), MakeXlaPrimitiveType(*dtype, /*device=*/nullptr));
  }
  return input.xla_shape();
}

}  // namespace

CumProd::CumProd(const XlaValue& input, int64_t dim,
                 c10::optional<at::ScalarType> dtype)
    : XlaNode(torch::lazy::OpKind(at::aten::cumprod), {input},
              [&]() { return NodeOutputShape(input, dtype); },
              /*num_outputs=*/1,
              torch::lazy::MHash(dim, torch::lazy::OptionalOr<int>(dtype, -1))),
      dim_(dim),
      dtype_(dtype) {}

torch::lazy::NodePtr CumProd::Clone(OpList operands) const {
  return torch::lazy::MakeNode<CumProd>(operands.at(0), dim_, dtype_);
}

XlaOpVector CumProd::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(LowerCumProd(input, dim_, dtype_), loctx);
}

std::string CumProd::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", dim=" << dim_;
  if (dtype_) {
    ss << ", dtype=" << *dtype_;
  }
  return ss.str();
}

}  // namespace torch_xla
