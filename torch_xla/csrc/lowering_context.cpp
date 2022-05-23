#include "torch_xla/csrc/lowering_context.h"

#include <sstream>
#include <stdexcept>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "torch/csrc/lazy/core/ir_metadata.h"

#include <iostream>

namespace torch_xla {
namespace ir {
namespace {

class HloMetadataSetter {
 public:
  HloMetadataSetter(LoweringContext* loctx, const torch::lazy::Node* node) {
    if (ShouldPopulateXlaOpMetadata()) {
      PopulateXlaOpMetadata(loctx, node);
      loctx_ = loctx;
    }
  }

  ~HloMetadataSetter() {
    if (loctx_ != nullptr) {
      loctx_->builder()->ClearOpMetadata();
    }
  }

 private:
  static bool ShouldPopulateXlaOpMetadata() {
    static bool op_metadata = xla::sys_util::GetEnvBool("XLA_HLO_DEBUG", false);
    return op_metadata;
  }

  static void PopulateXlaOpMetadata(LoweringContext* loctx,
                                    const torch::lazy::Node* node) {
    xla::OpMetadata metadata;
    // NOTE: we apply some string manipulation as xprof backend utility
    // for nesting/grouping traces depends on certain op name/type
    // patterns for classification.
    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/utils/tf_op_utils.cc#L55
    std::string op_type =
        absl::StrReplaceAll(node->op().ToString(), {{":", "_"}});
    metadata.set_op_type(op_type);
    const torch::lazy::MetaData& nmeta = node->metadata();
    std::string op_name_prefix;
    if (!nmeta.scope.empty()) {
      op_name_prefix =
          absl::StrCat(absl::StrReplaceAll(nmeta.scope, {{":", "_"}}), "/");
    }
    metadata.set_op_name(absl::StrCat(op_name_prefix, op_type));

    if (!nmeta.frame_info.empty()) {
      const torch::lazy::SourceLocation& frame = nmeta.frame_info.front();
      std::string::size_type pos = frame.file.find_last_of('/');
      if (pos == std::string::npos) {
        pos = 0;
      } else {
        ++pos;
      }
      metadata.set_source_file(frame.function + "@" + frame.file.substr(pos));
      metadata.set_source_line(frame.line);
    }
    loctx->builder()->SetOpMetadata(std::move(metadata));
  }

  LoweringContext* loctx_ = nullptr;
};

}  // namespace

LoweringContext::LoweringContext(const std::string& name,
                                 torch::lazy::BackendDevice device)
    : builder_(name), device_(std::move(device)) {
    std::cout << "[FTXJ LOG] LoweringContext::LoweringContext End" << name << std::endl;
}

LoweringContext::LoweringContext(
    const std::string& name, torch::lazy::BackendDevice device,
    absl::Span<const torch::lazy::Node* const> post_order,
    torch::lazy::Util::EmissionMap emit_status)
    : builder_(name),
      device_(std::move(device)),
      emit_status_(std::move(emit_status)) {
  std::cout << "[FTXJ LOG] LoweringContext::LoweringContext" << name << std::endl;
  for (auto node : post_order) {
    std::cout << "[FTXJ LOG] LoweringContext::LoweringContext call LowerNode" << name << std::endl;
    LowerNode(node);
  }
  std::cout << "[FTXJ LOG] LoweringContext::LoweringContext End" << name << std::endl;
}

xla::XlaOp LoweringContext::GetParameter(
    const std::shared_ptr<xla::ComputationClient::Data>& data) {
  std::cout << "[FTXJ LOG] LoweringContext::GetParameter.d from ComputationClient::Data" << std::endl;
  std::cout << "[FTXJ LOG] LoweringContext::GetParameter.d call ComputationClient:GetOpaqueHandle" << std::endl;
  xla::ComputationClient::Data::OpaqueHandle handle = data->GetOpaqueHandle();
  auto it = parameters_map_.find(handle);
  if (it == parameters_map_.end()) {
    std::cout << "[FTXJ LOG] LoweringContext::GetParameter.d not find param, build a new parameter" << std::endl;
    xla::XlaOp param =
        xla::Parameter(builder(), parameters_.size(), data->shape(),
                       absl::StrCat("p", parameters_.size()));
    it = parameters_map_.emplace(handle, Parameter{param, parameters_.size()})
             .first;
    parameters_.push_back(data);
  }
  parameter_sequence_.push_back(it->second.index);
  std::cout << "[FTXJ LOG] LoweringContext::GetParameter.d End" << std::endl;
  return it->second.param;
}

const std::vector<xla::ComputationClient::DataPtr>&
LoweringContext::GetParametersData() const {
  std::cout << "[FTXJ LOG] LoweringContext::GetParametersData End" << std::endl;
  return parameters_;
}

const std::vector<size_t>& LoweringContext::GetParameterSequence() const {
  std::cout << "[FTXJ LOG] LoweringContext::GetParameterSequence End" << std::endl;
  return parameter_sequence_;
}

size_t LoweringContext::AddResult(xla::XlaOp op) {
  std::cout << "[FTXJ LOG] LoweringContext::AddResult End" << std::endl;
  root_tuple_.push_back(std::move(op));
  return root_tuple_.size() - 1;
}

xla::XlaOp LoweringContext::GetResult(size_t index) const {
  std::cout << "[FTXJ LOG] LoweringContext::GetResult End" << std::endl;
  return root_tuple_.at(index);
}

void LoweringContext::SetResult(size_t index, xla::XlaOp op) {
  std::cout << "[FTXJ LOG] LoweringContext::SetResult assign op2root" << std::endl;
  root_tuple_.at(index) = std::move(op);
  std::cout << "[FTXJ LOG] LoweringContext::SetResult End" << std::endl;
}

xla::StatusOr<xla::XlaComputation> LoweringContext::Build() {
  std::cout << "[FTXJ LOG] LoweringContext::Build" << std::endl;
  std::cout << "[FTXJ MSG] build xla::XlaComputation from nodes" << std::endl;
  if (!root_tuple_.empty()) {
    std::cout << "[FTXJ LOG] LoweringContext::Build from root" << std::endl;
    xla::XlaOp root = xla::Tuple(builder(), root_tuple_);
    auto tmp = builder()->Build(root);
    std::cout << "[FTXJ LOG] LoweringContext::Build End" << std::endl;
    return tmp;
  }
  std::cout << "[FTXJ LOG] LoweringContext::Build End" << std::endl;
  auto tmp = builder()->Build();
  return tmp;
}

xla::StatusOr<xla::XlaComputation> LoweringContext::Build(xla::XlaOp root) {
  XLA_CHECK(root_tuple_.empty());
  std::cout << "[FTXJ LOG] LoweringContext::Build.r" << std::endl;
  std::cout << "[FTXJ MSG] build xla::XlaComputation from nodes" << std::endl;
  auto tmp = builder()->Build(root);
  std::cout << "[FTXJ LOG] LoweringContext::Build.r End" << std::endl;
  return tmp;
}

void LoweringContext::AssignOutputOp(const torch::lazy::Output& output,
                                     xla::XlaOp op) {
  std::cout << "[FTXJ LOG] LoweringContext::AssignOutputOp assign output->op pointer" << std::endl;
  emitted_outputs_[output] = std::move(op);
  std::cout << "[FTXJ LOG] LoweringContext::AssignOutputOp End" << std::endl;
}

xla::XlaOp LoweringContext::GetOutputOp(const torch::lazy::Output& output) {
  std::cout << "[FTXJ LOG] LoweringContext::GetOutputOp" << std::endl;
  std::cout << "find output in emitted_outputs_ context, if not, build post_order, and lower, then get" << std::endl;
  
  auto it = emitted_outputs_.find(output);
  if (it == emitted_outputs_.end()) {
    std::cout << "[FTXJ LOG] LoweringContext::GetOutputOp find in context fail" << std::endl;
    std::cout << "[FTXJ LOG] LoweringContext::GetOutputOp call ComputePostOrder" << std::endl;
    auto post_order = Util::ComputePostOrder(output.node, &emit_status_);
    for (auto node : post_order) {
      std::cout << "[FTXJ LOG] LoweringContext::GetOutputOp call LowerNode" << std::endl;
      LowerNode(node);
    }
    // At this point the outpout better be present, otherwise there is an issue
    // with the lowering code.
    it = emitted_outputs_.find(output);
    XLA_CHECK(it != emitted_outputs_.end())
        << "No XLA operation emitted for output: " << output;
  }
  std::cout << "[FTXJ LOG] LoweringContext::GetOutputOp End" << std::endl;
  return it->second;
}

XlaOpVector LoweringContext::LowerNode(const torch::lazy::Node* node) {
  XlaOpVector result_ops;
  std::cout << "[FTXJ LOG] LoweringContext::LowerNode. cast (Lazy node) -> (XlaOpVector) and do lower " << std::endl;
  try {
    HloMetadataSetter meta_setter(this, node);
    const XlaNode* casted = dynamic_cast<const XlaNode*>(node);
    std::cout << "[FTXJ LOG] LoweringContext::LowerNode call Lower" << std::endl;
    result_ops = casted->Lower(this);
  } catch (const std::exception& ex) {
    ReportBuilderError(node, ex.what());
  }
  if (!builder()->first_error().ok()) {
    ReportBuilderError(node, /*error_msg=*/nullptr);
  }
  std::cout << "[FTXJ LOG] LoweringContext::LowerNode End" << std::endl;
  return result_ops;
}

void LoweringContext::ReportBuilderError(const torch::lazy::Node* node,
                                         const char* error_msg) {
  std::cout << "[FTXJ ERROR]" << std::endl;
  std::stringstream ss;
  ss << "Error while lowering: " << node->ToString() << "\n";
  if (!builder()->first_error().ok()) {
    ss << "XLA builder error: " << builder()->GetCurrentStatus() << "\n";
  }
  if (error_msg != nullptr) {
    ss << "Error: " << error_msg << "\n";
  }
  const torch::lazy::MetaData& nmeta = node->metadata();
  if (!nmeta.scope.empty()) {
    ss << "Scope: " << nmeta.scope << "\n";
  }
  ss << nmeta.frame_info;
  throw std::runtime_error(ss.str());
}

}  // namespace ir
}  // namespace torch_xla
