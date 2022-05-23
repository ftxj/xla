#include "torch_xla/csrc/aten_cpu_fallback.h"

#include <tensorflow/compiler/xla/xla_client/debug_macros.h>
#include <tensorflow/compiler/xla/xla_client/metrics.h>
#include <tensorflow/compiler/xla/xla_client/tf_logging.h>
#include <torch_xla/csrc/function_call_tracker.h>

#include <unordered_map>
#include <iostream>

namespace torch_xla {

static std::unordered_map<std::string, ::xla::metrics::Counter*>
    _cpu_fallback_counters;

void xla_cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  XLA_FN_TRACK(3);
  std::cout << "[FTXJ LOG] xla_cpu_fallback" << std::endl; 
  const auto name = c10::toString(op.operator_name());
  std::cout << "\t[op name] " << name << std::endl;
  // Manually applying the XLA_COUNTER macro.
  // We need to do it ourselves and explicitly keep a mapping of counters
  // because this boxed fallback kernel is used by multiple operators,
  // and the macro stamps out a static Counter object with a fixed name
  // at the code location that it was called.
  if (_cpu_fallback_counters.find(name) == _cpu_fallback_counters.end()) {
    _cpu_fallback_counters[name] = new ::xla::metrics::Counter(name);
  }
  _cpu_fallback_counters[name]->AddValue(1);

  auto& args = op.schema().arguments();
  auto arguments = torch::jit::last(stack, args.size());

  // Log each tensor argument.
  for (int64_t idx = 0; idx < arguments.size(); ++idx) {
    const auto& ivalue = arguments[idx];
    if (ivalue.isTensor()) {
      TF_VLOG(3) << ivalue.toTensor().toString();
    }
  }

  // Call the actual boxed CPU fallback.
  std::cout << "[FTXJ LOG] xla_cpu_fallback call at::native::cpu_fallback" << std::endl;
  at::native::cpu_fallback(op, stack);
  std::cout << "[FTXJ LOG] xla_cpu_fallback End" << std::endl;
}

TORCH_LIBRARY_IMPL(_, XLA, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&xla_cpu_fallback>());
}

}  // namespace torch_xla
