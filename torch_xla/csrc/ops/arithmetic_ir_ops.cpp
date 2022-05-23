#include "torch_xla/csrc/ops/arithmetic_ir_ops.h"

#include <memory>

#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/ops.h"

#include <iostream>

namespace torch_xla {
namespace ir {

torch::lazy::NodePtr operator+(const XlaValue& node1, const XlaValue& node2) {
  std::cout << "[FTXJ LOG] NodePtr::Add Constructor" << std::endl;
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    std::cout << "[FTXJ LOG] NodePtr::Add Constructor lower_fn" << std::endl;
    std::cout << "[FTXJ LOG] NodePtr::Add Constructor lower_fn call GetOutputOp Op0" << std::endl;
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    std::cout << "[FTXJ LOG] NodePtr::Add Constructor lower_fn call GetOutputOp Op1" << std::endl;
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    auto tmp = node.ReturnOp(XlaHelpers::PromotedAdd(op0, op1), loctx);
    return tmp;
  };
  auto tmp = ops::GenericOp(torch::lazy::OpKind(at::aten::add), {node1, node2},
                        XlaHelpers::GetPromotedBinaryOpShape(node1.xla_shape(),
                                                             node2.xla_shape()),
                        std::move(lower_fn));
  return tmp;
}

torch::lazy::NodePtr operator-(const XlaValue& node1, const XlaValue& node2) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(XlaHelpers::PromotedSub(op0, op1), loctx);
  };
  return ops::GenericOp(torch::lazy::OpKind(at::aten::sub), {node1, node2},
                        XlaHelpers::GetPromotedBinaryOpShape(node1.xla_shape(),
                                                             node2.xla_shape()),
                        std::move(lower_fn));
}

torch::lazy::NodePtr operator*(const XlaValue& node1, const XlaValue& node2) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(XlaHelpers::PromotedMul(op0, op1), loctx);
  };
  return ops::GenericOp(torch::lazy::OpKind(at::aten::mul), {node1, node2},
                        XlaHelpers::GetPromotedBinaryOpShape(node1.xla_shape(),
                                                             node2.xla_shape()),
                        std::move(lower_fn));
}

torch::lazy::NodePtr operator/(const XlaValue& node1, const XlaValue& node2) {
  auto lower_fn = [](const XlaNode& node,
                     LoweringContext* loctx) -> XlaOpVector {
    xla::XlaOp op0 = loctx->GetOutputOp(node.operand(0));
    xla::XlaOp op1 = loctx->GetOutputOp(node.operand(1));
    return node.ReturnOp(XlaHelpers::PromotedDiv(op0, op1), loctx);
  };
  return ops::GenericOp(torch::lazy::OpKind(at::aten::div), {node1, node2},
                        XlaHelpers::GetPromotedBinaryOpShape(node1.xla_shape(),
                                                             node2.xla_shape()),
                        std::move(lower_fn));
}

}  // namespace ir
}  // namespace torch_xla
