#include "Dummy.h"
#include "../Model.h"
#include "../Tensor.h"

Dummy::Dummy(SimulationConfig config, Model* model, onnx::NodeProto& node_proto, uint32_t target_core)
    : Operation(config, model, node_proto, target_core) {
  std::vector<uint32_t> new_dim;
  new_dim = {64, 4096};
  _input_shape = get_input(0)->get_dims();
  if((node_proto.op_type() == "Gather") && (_input_shape.at(0) == 32000)){
    get_input(0)->resize_tensor(new_dim);
  }

  _input_shape = get_input(0)->get_dims();
  //std::cout << _input_shape.at(0) << std::endl;
  _output_shape = _input_shape;
  spdlog::info("output_shape : {}", _output_shape);
  spdlog::info("output name : {} {}", node_proto.output(0).c_str());

  for (int i=0;i<node_proto.output().size();i++) {
    Tensor* pre_defind_tensor = _model->find_tensor(node_proto.output(i));
    if (pre_defind_tensor == nullptr) {
      std::unique_ptr<Tensor> output_tensor = std::make_unique<Tensor>(
          _id, node_proto.output(i), _output_shape, _config.precision, false);
      _outputs.push_back(output_tensor.get()->get_id());
      _model->add_tensor(std::move(output_tensor));
    } else {
      pre_defind_tensor->redefine_tensor(_id, _output_shape);
    }
  }
}

void Dummy::initialize_tiles(MappingTable& mapping_table) {
  std::unique_ptr<Tile> tile = std::make_unique<Tile>(Tile{
                        .status = Tile::Status::INITIALIZED,
                        .optype="Dummy",
                        .layer_id=_id,
                        .skip = true});
  _tiles.push_back(std::move(tile));
  initialize_instructions(_tiles.back().get(), Mapping{});
}

void Dummy::initialize_instructions(Tile* tile, Mapping mapping) {
}