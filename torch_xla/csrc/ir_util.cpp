#include "torch_xla/csrc/ir_util.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include <iostream>

namespace torch_xla {
namespace ir {

std::vector<const torch::lazy::Node*> Util::ComputePostOrder(
    const torch::lazy::Node* node, EmissionMap* emap) {
  std::cout << "[FTXJ LOG] Util::ComputePostOrder" << std::endl;
  std::vector<const torch::lazy::Node*> post_order;
  std::vector<const torch::lazy::Node*> queue;
  queue.push_back(node);
  while (!queue.empty()) {
    node = queue.back();

    std::string node_string = node->ToString();
    std::string op_string = node->op().ToString();
    
    std::cout << "traversal node(str=" << node_string << ", op=" << op_string << ")\n";

    auto it = emap->find(node);

    if (it == emap->end()) {
      std::cout << "node first traversal" << std::endl;
      (*emap)[node] = torch::lazy::Util::kEmitting;
      for (auto& output : node->operands()) {
        std::cout << "operands --- node(str=" << output.node->ToString() 
          << ", op=" << output.node->op().ToString() << ")\n";
        
        auto oit = emap->find(output.node);
        if (oit == emap->end()) {
          std::cout << "first traversal this node, push into queue" << std::endl;
          queue.push_back(output.node);
        } else if (oit->second == torch::lazy::Util::kEmitting) {
          XLA_ERROR() << "Graph loop found at " << *output.node;
        }
      }
    } else if (it->second == torch::lazy::Util::kEmitting) {
      std::cout << "node second traversal" << std::endl;
      for (auto& output : node->operands()) {
        std::cout << "operands --- node(str=" << output.node->ToString() 
          << ", op=" << output.node->op().ToString() << ")\n";
        auto oit = emap->find(output.node);
        XLA_CHECK(oit != emap->end() &&
                  oit->second == torch::lazy::Util::kEmitted)
            << "Graph loop found at " << *output.node;
      }
      (*emap)[node] = torch::lazy::Util::kEmitted;
      std::cout << "node into post order" << std::endl;
      post_order.push_back(node);
      queue.pop_back();
    } else {
      XLA_CHECK_EQ(it->second, torch::lazy::Util::kEmitted);
      queue.pop_back();
    }
  }
  std::cout << "\nAfter ComputePostOrder, return data is" << std::endl;
  for(auto node : post_order) {
    std::string node_string = node->ToString();
    std::string op_string = node->op().ToString();
    std::cout << "node(str=" << node_string << ", op=" << op_string << ")\n";
  }
  std::cout << std::endl;
  return post_order;
}

std::vector<const torch::lazy::Node*> Util::ComputePostOrder(
    absl::Span<const torch::lazy::Node* const> nodes, EmissionMap* emap) {
  std::vector<const torch::lazy::Node*> post_order;
  for (auto node : nodes) {
    std::string node_string = node->ToString();
    std::string op_string = node->op().ToString();
    std::cout << "call ComputePostOrder on node(str=" << node_string << ", op=" << op_string << ")\n";
    auto node_post_order = ComputePostOrder(node, emap);
    post_order.insert(post_order.end(), node_post_order.begin(),
                      node_post_order.end());
    std::cout << "\nafter call ComputePostOrder, post order is" << std::endl;
    for(auto node : post_order) {
      std::string node_string = node->ToString();
      std::string op_string = node->op().ToString();
      std::cout << "node(str=" << node_string << ", op=" << op_string << ")\n";
    }
  }
  return post_order;
}

std::vector<const torch::lazy::Node*> Util::ComputePostOrder(
    absl::Span<const torch::lazy::Node* const> nodes) {
  EmissionMap emap;
  return ComputePostOrder(nodes, &emap);
}

std::vector<XlaValue> Util::Clone(
    absl::Span<const XlaValue> values,
    absl::Span<const torch::lazy::Node* const> post_order) {
  
  std::cout << "[FRXJ LOG] Util::Clone.v.p" << std::endl;

  std::unordered_map<const torch::lazy::Node*, torch::lazy::NodePtr> clone_map;
  for (auto node : post_order) {
    if (clone_map.count(node) > 0) {
      continue;
    }
    std::vector<XlaValue> inputs;
    for (auto& output : node->operands()) {
      auto it = clone_map.find(output.node);
      XLA_CHECK(it != clone_map.end())
          << "Bad post-order: " << node->ToString();
      inputs.emplace_back(it->second, output.index);
    }
    const XlaNode* casted = dynamic_cast<const XlaNode*>(node);
    clone_map[node] = casted->Clone(inputs);
  }

  std::vector<XlaValue> cloned;
  for (auto& value : values) {
    auto it = clone_map.find(value.node.get());
    XLA_CHECK(it != clone_map.end()) << "Bad post-order: " << value->ToString();
    cloned.emplace_back(it->second, value.index);
  }

  std::cout << "[FRXJ LOG] Util::Clone.v.p End" << std::endl;

  return cloned;
}

std::vector<XlaValue> Util::Clone(absl::Span<const XlaValue> values) {
  std::cout << "[FRXJ LOG] Util::Clone.v" << std::endl;
  std::vector<const torch::lazy::Node*> nodes;
  for (auto& value : values) {
    nodes.push_back(value.node.get());
  }
  std::cout << "[FRXJ LOG] Util::Clone.v call ComputePostOrder" << std::endl;
  std::vector<const torch::lazy::Node*> post_order = ComputePostOrder(nodes);
  std::cout << "[FRXJ LOG] Util::Clone.v call Clone.v.p" << std::endl;
  auto tmp = Clone(values, post_order);
  std::cout << "[FRXJ LOG] Util::Clone.v End" << std::endl;
  return tmp;
}

size_t Util::GetGraphSize(absl::Span<const torch::lazy::Node* const> nodes) {
  std::cout << "[FRXJ LOG] Util::GetGraphSize" << std::endl;
  std::cout << "[FRXJ LOG] Util::GetGraphSize call ComputePostOrder" << std::endl;
  std::vector<const torch::lazy::Node*> post_order = ComputePostOrder(nodes);
  std::cout << "[FRXJ LOG] Util::GetGraphSize" << std::endl;
  return post_order.size();
}

}  // namespace ir
}  // namespace torch_xla
