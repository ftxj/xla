#include "torch_xla/csrc/ops/adam_optimizer_step.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const XlaValue& step, const XlaValue& param) {
  return xla::ShapeUtil::MakeTupleShape(
      {/*step=*/step.xla_shape(), /*param=*/param.xla_shape(),
       /*exp_avg=*/param.xla_shape(), /*exp_avg_sq=*/param.xla_shape(),
       /*max_exp_avg_sq=*/param.xla_shape()});
}

}  // namespace

AdamOptimizerStep::AdamOptimizerStep(
    const XlaValue& found_inf, const XlaValue& step, const XlaValue& param,
    const XlaValue& grad, const XlaValue& exp_avg, const XlaValue& exp_avg_sq,
    const XlaValue& max_exp_avg_sq, const XlaValue& beta1,
    const XlaValue& beta2, const XlaValue& lr, const XlaValue& weight_decay,
    const XlaValue& eps, bool use_weight_decay, bool use_amsgrad,
    bool use_adamw)
    : XlaNode(xla_adam_optimizer_step,
              {found_inf, step, param, grad, exp_avg, exp_avg_sq,
               max_exp_avg_sq, beta1, beta2, lr, weight_decay, eps},
              NodeOutputShape(step, param),
              /*num_outputs=*/5,
              torch::lazy::MHash(use_weight_decay, use_amsgrad, use_adamw)),
      use_weight_decay_(use_weight_decay),
      use_amsgrad_(use_amsgrad),
      use_adamw_(use_adamw) {}

torch::lazy::NodePtr AdamOptimizerStep::Clone(OpList operands) const {
  return ir::MakeNode<AdamOptimizerStep>(
      operands.at(0), operands.at(1), operands.at(2), operands.at(3),
      operands.at(4), operands.at(5), operands.at(6), operands.at(7),
      operands.at(8), operands.at(9), operands.at(10), operands.at(11),
      use_weight_decay_, use_amsgrad_, use_adamw_);
}

XlaOpVector AdamOptimizerStep::Lower(LoweringContext* loctx) const {
  xla::XlaOp found_inf = loctx->GetOutputOp(operand(0));
  xla::XlaOp step = loctx->GetOutputOp(operand(1));
  xla::XlaOp param = loctx->GetOutputOp(operand(2));
  xla::XlaOp grad = loctx->GetOutputOp(operand(3));
  xla::XlaOp exp_avg = loctx->GetOutputOp(operand(4));
  xla::XlaOp exp_avg_sq = loctx->GetOutputOp(operand(5));
  xla::XlaOp max_exp_avg_sq = loctx->GetOutputOp(operand(6));
  xla::XlaOp beta1 = loctx->GetOutputOp(operand(7));
  xla::XlaOp beta2 = loctx->GetOutputOp(operand(8));
  xla::XlaOp lr = loctx->GetOutputOp(operand(9));
  xla::XlaOp weight_decay = loctx->GetOutputOp(operand(10));
  xla::XlaOp eps = loctx->GetOutputOp(operand(11));
  return ReturnOps(
      BuildAdamOptimizerStep(found_inf, step, param, grad, exp_avg, exp_avg_sq,
                             max_exp_avg_sq, beta1, beta2, lr, weight_decay,
                             eps, use_weight_decay_, use_amsgrad_, use_adamw_),
      loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
