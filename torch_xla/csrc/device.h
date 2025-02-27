#pragma once

#include <iostream>
#include <string>

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch/csrc/lazy/backend/backend_device.h"
#include "torch/csrc/lazy/core/hash.h"
#include "torch/csrc/lazy/core/util.h"

#include <iostream>

namespace torch_xla {

enum class XlaDeviceType { CPU, GPU, TPU };

struct DeviceType : public torch::lazy::BackendDeviceType {
  DeviceType() { 
    std::cout << "[FTXJ LOG] " << "DeviceType default construction CPU" << std::endl;
    type = static_cast<int>(XlaDeviceType::CPU); 
  }
  
  DeviceType(XlaDeviceType xla_device_type) {
    if(xla_device_type == XlaDeviceType::GPU)
      std::cout << "[FTXJ LOG] " << "DeviceType construction GPU" << std::endl;
    if(xla_device_type == XlaDeviceType::CPU)
      std::cout << "[FTXJ LOG] " << "DeviceType construction CPU" << std::endl;
    if(xla_device_type == XlaDeviceType::TPU)
      std::cout << "[FTXJ LOG] " << "DeviceType construction TPU" << std::endl;
    type = static_cast<int>(xla_device_type);
  }

  std::string toString() const override;
};

torch::lazy::BackendDevice ParseDeviceString(const std::string& device_spec);

const torch::lazy::BackendDevice* GetDefaultDevice();

torch::lazy::BackendDevice GetCurrentDevice();

torch::lazy::BackendDevice SetCurrentDevice(
    const torch::lazy::BackendDevice& device);

static inline torch::lazy::BackendDevice GetDeviceOrCurrent(
    const torch::lazy::BackendDevice* device) {
  return device != nullptr ? *device : GetCurrentDevice();
}

}  // namespace torch_xla
