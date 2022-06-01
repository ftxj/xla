#include "torch_xla/csrc/ops/infer_output_shape.h"

#include "torch_xla/csrc/helpers.h"
#include <iostream>

namespace torch_xla {
namespace ir {
namespace ops {

xla::Shape InferOutputShape(absl::Span<const xla::Shape> input_shapes,
                            const LowerForShapeFn& core_lowering_fn) {
  std::cout << "[FTXJ LOG] InferOutputShape " << std::endl;
  
  std::cout << "[FTXJ LOG] XlaBuilder InferOutputShape" << std::endl;
  xla::XlaBuilder b("InferOutputShape");

  std::vector<xla::XlaOp> parameters;
  for (size_t parameter_number = 0; parameter_number < input_shapes.size();
       ++parameter_number) {
    auto tmp_s = xla::ShapeUtil::HumanString(input_shapes[parameter_number]);
    
    std::cout << "construct xla::Parameter id=p" << 
      parameter_number << ", value=" << tmp_s << std::endl;

    parameters.push_back(xla::Parameter(&b, parameter_number,
                                        input_shapes[parameter_number],
                                        absl::StrCat("p", parameter_number)));
  }
  xla::XlaOp result = core_lowering_fn(parameters);

  auto tmp = XlaHelpers::ShapeOfXlaOp(result);
  auto tmp_s = xla::ShapeUtil::HumanString(tmp);
  std::cout << "OutputShape is = " << tmp_s << std::endl;
  
  std::cout << "[FTXJ LOG] InferOutputShape End" << std::endl;
  return tmp;
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
