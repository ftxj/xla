#include "torch_xla/csrc/aten_xla_bridge.h"

#include <map>
#include <string>
#include <vector>
#include <iostream>
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/tensor_impl.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {
namespace bridge {
namespace {

class AtenXlaDeviceMapper {
 public:
  static AtenXlaDeviceMapper* Get();

  size_t GetDeviceOrdinal(const torch::lazy::BackendDevice& device) const {
    auto it = devices_ordinals_.find(device);
    XLA_CHECK(it != devices_ordinals_.end()) << device;
    return it->second;
  }

  const torch::lazy::BackendDevice& GetDeviceFromOrdinal(size_t ordinal) const {
    std::cout << "[GetDeviceFromOrdinal] from device index to lazy::BackendDevice" << std::endl;
    return devices_.at(ordinal);
  }

 private:
  AtenXlaDeviceMapper() {
    std::cout << "[AtenXlaDeviceMapper] init. need call xla::ComputationClient Get()" << std::endl;
    for (auto& device_str : xla::ComputationClient::Get()->GetLocalDevices()) {
      devices_.emplace_back(ParseDeviceString(device_str));
      devices_ordinals_[devices_.back()] = devices_.size() - 1;
    }
  }

  std::vector<torch::lazy::BackendDevice> devices_;
  std::map<torch::lazy::BackendDevice, size_t> devices_ordinals_;
};

AtenXlaDeviceMapper* AtenXlaDeviceMapper::Get() {
  std::cout << "[AtenXlaDeviceMapper::Get]" << std::endl;
  static AtenXlaDeviceMapper* device_mapper = new AtenXlaDeviceMapper();
  return device_mapper;
}

XLATensorImpl* GetXlaTensorImpl(const at::Tensor& tensor) {
  std::cout << "[GetXlaTensorImpl] convert from at::Tensor -> XLATensorImpl" << std::endl;
  return dynamic_cast<XLATensorImpl*>(tensor.unsafeGetTensorImpl());
}

}  // namespace

c10::optional<XLATensor> TryGetXlaTensor(const at::Tensor& tensor) {
  std::cout << "[TryGetXlaTensor] convert from at::Tensor -> option<XLATensor>" << std::endl;
  XLATensorImpl* impl = GetXlaTensorImpl(tensor);
  if (impl == nullptr) {
    return c10::nullopt;
  }
  return impl->tensor();
}

bool IsXlaTensor(const at::Tensor& tensor) {
  std::cout << "[IsXlaTensor] convert from at::Tensor -> bool" << std::endl;
  return GetXlaTensorImpl(tensor) != nullptr;
}

XLATensor GetXlaTensor(const at::Tensor& tensor) {
  std::cout << "[GetXlaTensor] convert from at::Tensor -> XLATensor" << std::endl;
  auto xtensor = TryGetXlaTensor(tensor);
  XLA_CHECK(xtensor) << "Input tensor is not an XLA tensor: "
                     << tensor.toString();
  return *xtensor;
}

void ReplaceXlaTensor(const at::Tensor& tensor, XLATensor new_xla_tensor) {
  std::cout << "[ReplaceXlaTensor] convert from at::Tensor -> XLATensor" << std::endl;
  XLATensorImpl* impl =
      dynamic_cast<XLATensorImpl*>(tensor.unsafeGetTensorImpl());
  XLA_CHECK(impl != nullptr)
      << "Input tensor is not an XLA tensor: " << tensor.toString();
  impl->set_tensor(std::move(new_xla_tensor));
}

std::vector<XLATensor> GetXlaTensors(absl::Span<const at::Tensor> tensors) {
  std::cout << "[GetXlaTensors] convert from Span(at::Tensor) -> vector(XLATensor)" << std::endl;
  std::vector<XLATensor> xla_tensors;
  xla_tensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    xla_tensors.push_back(bridge::GetXlaTensor(tensor));
  }
  return xla_tensors;
}

XLATensor GetOrCreateXlaTensor(const at::Tensor& tensor,
                               const torch::lazy::BackendDevice& device) {
  std::cout << "[GetOrCreateXlaTensor] convert at::Tensor & Device -> XLATensor" << std::endl;
  if (!tensor.defined()) {
    return XLATensor();
  }
  auto xtensor = TryGetXlaTensor(tensor);
  return xtensor ? *xtensor : XLATensor::Create(tensor, device);
}

XLATensor GetOrCreateXlaTensor(const c10::optional<at::Tensor>& tensor,
                               const torch::lazy::BackendDevice& device) {
  std::cout << "[GetOrCreateXlaTensor] convert option<at::Tensor> & Device -> XLATensor" << std::endl;
  if (!IsDefined(tensor)) {
    return XLATensor();
  }
  auto xtensor = TryGetXlaTensor(*tensor);
  return xtensor ? *xtensor : XLATensor::Create(*tensor, device);
}

std::vector<XLATensor> GetOrCreateXlaTensors(
    absl::Span<const at::Tensor> tensors,
    const torch::lazy::BackendDevice& device) {
      std::cout << "[GetOrCreateXlaTensors]" << std::endl;
  std::vector<XLATensor> xla_tensors;
  for (const at::Tensor& tensor : tensors) {
    xla_tensors.push_back(bridge::GetOrCreateXlaTensor(tensor, device));
  }
  return xla_tensors;
}

std::vector<at::Tensor> XlaCreateTensorList(const at::TensorList& tensors) {
  std::cout << "[XlaCreateTensorList] convert at::TensorList -> vector<XLATensor>" << std::endl;
  std::vector<at::Tensor> aten_xla_tensors(tensors.size());
  std::vector<XLATensor> xla_tensors;
  // We need to separate out the defined tensors first, GetXlaTensor() doesn't
  // work with undefined tensors.
  std::vector<bool> to_translate(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    const at::Tensor& tensor = tensors[i];
    if (tensor.defined()) {
      auto xtensor = TryGetXlaTensor(tensor);
      if (xtensor) {
        to_translate[i] = true;
        xla_tensors.push_back(*xtensor);
      } else {
        aten_xla_tensors[i] = tensor;
      }
    }
  }
  auto defined_aten_xla_tensors = XLATensor::GetTensors(&xla_tensors);
  // Insert undefined tensors into the result, back into the original undefined
  // positions.
  for (size_t i = 0, defined_pos = 0; i < tensors.size(); ++i) {
    if (to_translate[i]) {
      aten_xla_tensors[i] = std::move(defined_aten_xla_tensors[defined_pos++]);
    }
  }
  return aten_xla_tensors;
}

std::vector<c10::optional<at::Tensor>> XlaCreateOptTensorList(
    const std::vector<c10::optional<at::Tensor>>& tensors) {
  std::cout << "[XlaCreateOptTensorList]" << std::endl;
  std::vector<c10::optional<at::Tensor>> opt_aten_xla_tensors(tensors.size());
  std::vector<at::Tensor> materialized_tensors;
  std::vector<bool> to_translate(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto tensor = tensors[i];
    if (tensor.has_value()) {
      to_translate[i] = true;
      materialized_tensors.push_back(*tensor);
    }
  }
  auto aten_materialzied_tensors = XlaCreateTensorList(materialized_tensors);
  for (size_t i = 0, defined_pos = 0; i < tensors.size(); ++i) {
    if (to_translate[i]) {
      opt_aten_xla_tensors[i] =
          std::move(aten_materialzied_tensors[defined_pos++]);
    }
  }
  return opt_aten_xla_tensors;
}

void XlaUpdateTensors(absl::Span<const at::Tensor> dest_xla_tensors,
                      absl::Span<const at::Tensor> source_cpu_tensors,
                      absl::Span<const size_t> indices) {
    std::cout << "[XlaUpdateTensors]" << std::endl;
  for (auto index : indices) {
    at::Tensor dest = dest_xla_tensors.at(index);
    at::Tensor source = source_cpu_tensors.at(index);
    XLATensorImpl* dest_impl = GetXlaTensorImpl(dest);
    if (dest_impl != nullptr) {
      auto xla_source = TryGetXlaTensor(source);
      if (!xla_source) {
        dest_impl->tensor().UpdateFromTensorOut(source);
      } else {
        dest_impl->tensor().UpdateFromTensorOut(*xla_source);
      }
      dest_impl->force_refresh_sizes();
    } else {
      dest.resize_as_(source).copy_(source);
    }
  }
}

c10::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const at::Tensor& tensor) {
        std::cout << "[GetXlaDevice]" << std::endl;
  auto xtensor = TryGetXlaTensor(tensor);
  if (!xtensor) {
    return c10::nullopt;
  }
  return xtensor->GetDevice();
}

c10::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const c10::optional<at::Tensor>& tensor) {
      std::cout << "[GetXlaDevice]" << std::endl;
  if (!tensor.has_value()) {
    return c10::nullopt;
  }
  return GetXlaDevice(*tensor);
}

c10::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const at::TensorList& tensors) {
        std::cout << "[GetXlaDevice]" << std::endl;
  for (const auto& tensor : tensors) {
    auto device = GetXlaDevice(tensor);
    if (device) {
      return device;
    }
  }
  return c10::nullopt;
}

c10::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const at::TensorOptions& tensor_options) {
    std::cout << "[GetXlaDevice]" << std::endl;
  if (!tensor_options.has_device()) {
    return c10::nullopt;
  }
  return GetXlaDevice(tensor_options.device());
}

c10::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const c10::Device& device) {
    std::cout << "[GetXlaDevice]" << std::endl;
  if (device.type() != at::kXLA) {
    return c10::nullopt;
  }
  return AtenDeviceToXlaDevice(device);
}

c10::optional<torch::lazy::BackendDevice> GetXlaDevice(
    const c10::optional<c10::Device>& device) {
  std::cout << "[GetXlaDevice]" << std::endl;
  if (!device) {
    return c10::nullopt;
  }
  return GetXlaDevice(*device);
}

torch::lazy::BackendDevice AtenDeviceToXlaDevice(const c10::Device& device) {
  std::cout << "[AtenDeviceToXlaDevice] call. input = c10::Device, output = lazy::BackendDevice" << std::endl;
  XLA_CHECK_EQ(device.type(), at::kXLA) << device;
  int ordinal = device.has_index() ? device.index() : -1;
  if (ordinal < 0) {
    std::cout << "[AtenDeviceToXlaDevice] fall in index < 0" << std::endl;
    c10::Device current_device = GetCurrentAtenDevice();
    if (current_device.has_index()) {
      ordinal = current_device.index();
    }
  }
  if (ordinal < 0) {
    std::cout << "[AtenDeviceToXlaDevice] fall in index < 0 two" << std::endl;
    return GetCurrentDevice();
  }
  return AtenXlaDeviceMapper::Get()->GetDeviceFromOrdinal(ordinal);
}

c10::Device XlaDeviceToAtenDevice(const torch::lazy::BackendDevice& device) {
  std::cout << "[XlaDeviceToAtenDevice]" << std::endl;
  return c10::Device(at::kXLA,
                     AtenXlaDeviceMapper::Get()->GetDeviceOrdinal(device));
}

std::string ToXlaString(const c10::Device& device) {
  std::cout << "[ToXlaString]" << std::endl;
  return absl::StrCat("xla:", device.index());
}

c10::Device AtenDefaultDevice() {
  std::cout << "[AtenDefaultDevice]" << std::endl;
  return XlaDeviceToAtenDevice(*GetDefaultDevice());
}

c10::Device SetCurrentDevice(const c10::Device& device) {
  std::cout << "[bridge::SetCurrentDevice] input: c10::Device" << std::endl;
  torch::lazy::BackendDevice prev_device =
      torch_xla::SetCurrentDevice(AtenDeviceToXlaDevice(device));
  return XlaDeviceToAtenDevice(prev_device);
}

torch::lazy::BackendDevice SetCurrentDevice(
    const torch::lazy::BackendDevice& device) {
  std::cout << "[SetCurrentDevice]" << std::endl;
  return torch_xla::SetCurrentDevice(device);
}

c10::Device GetCurrentAtenDevice() {
  std::cout << "[GetCurrentAtenDevice]" << std::endl;
  return XlaDeviceToAtenDevice(torch_xla::GetCurrentDevice());
}

at::Tensor XlaToAtenTensor(XLATensor xla_tensor,
                           const at::TensorOptions& tensor_options) {
  std::cout << "[XlaToAtenTensor] XLATensor -> at::Tensor" << std::endl;
  if (tensor_options.has_device()) {
    XLA_CHECK_NE(tensor_options.device().type(), at::kXLA);
  }
  at::Tensor tensor = xla_tensor.ToTensor(/*detached=*/false);
  // We need to copy the tensor since it is cached within the XLATensor, and
  // returning it directly might expose it to in place changes. Which there was
  // COW option :)
  return tensor.to(tensor_options, /*non_blocking=*/false, /*copy=*/true);
}

at::Tensor AtenFromXlaTensor(XLATensor xla_tensor) {
  std::cout << "[AtenFromXlaTensor] XLATensor -> Tensor" << std::endl;
  return xla_tensor.is_null() ? at::Tensor()
                              : at::Tensor(c10::make_intrusive<XLATensorImpl>(
                                    std::move(xla_tensor)));
}

std::vector<at::Tensor> AtenFromXlaTensors(
    absl::Span<const XLATensor> xla_tensors) {
  std::cout << "[AtenFromXlaTensors] <XLATensor> -> <Tensor>" << std::endl;
  std::vector<at::Tensor> tensors;
  tensors.reserve(xla_tensors.size());
  for (auto& tensor : xla_tensors) {
    tensors.emplace_back(AtenFromXlaTensor(tensor));
  }
  return tensors;
}

at::Tensor CreateXlaTensor(
    at::Tensor tensor,
    const c10::optional<torch::lazy::BackendDevice>& device) {
  std::cout << "[CreateXlaTensors] return at::Tensor 2" << std::endl;
  if (tensor.defined() && device) {
    XLATensor xla_tensor = XLATensor::Create(std::move(tensor), *device);
    tensor = AtenFromXlaTensor(xla_tensor);
  }
  return tensor;
}

std::vector<at::Tensor> CreateXlaTensors(
    const std::vector<at::Tensor>& tensors,
    const c10::optional<torch::lazy::BackendDevice>& device) {
  std::cout << "[CreateXlaTensors] return at::Tensor" << std::endl;
  std::vector<at::Tensor> xtensors;
  for (auto& tensor : tensors) {
    xtensors.push_back(CreateXlaTensor(tensor, device));
  }
  return xtensors;
}

}  // namespace bridge
}  // namespace torch_xla

namespace torch {
namespace lazy {

torch_xla::XLATensor GetXlaTensorOrCreateForWrappedNumber(
    const at::Tensor& tensor, const torch::lazy::BackendDevice& device) {
  std::cout << "GetXlaTensorOrCreateForWrappedNumber" << std::endl;
  return (tensor.unsafeGetTensorImpl()->is_wrapped_number() ||
          (tensor.dim() == 0 && tensor.numel() == 1))
             ? torch_xla::bridge::GetOrCreateXlaTensor(tensor, device)
             : torch_xla::bridge::GetXlaTensor(tensor);
}

}  // namespace lazy
}  // namespace torch
