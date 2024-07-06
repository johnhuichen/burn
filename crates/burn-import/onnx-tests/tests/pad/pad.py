#!/usr/bin/env python3

# used to generate model: onnx-tests/tests/pad/pad.onnx

import onnx
from onnx import helper
from onnx import TensorProto


def main():
    input_tensor = helper.make_tensor_value_info(
        "input_tensor", TensorProto.FLOAT, [1, 2]
    )

    pads = helper.make_tensor_value_info("Pads", TensorProto.INT64, [4])

    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])

    node_def = helper.make_node(
        "Pad",
        name="/Pad",
        inputs=["input_tenor", "pads"],
        outputs=["output"],
        mode="constant",
    )

    # Create the graph (GraphProto)
    graph_def = helper.make_graph(
        nodes=[node_def],
        name="PadGraph",
        inputs=[input_tensor, pads],
        outputs=[output],
        # [
        #     helper.make_tensor(
        #         "Pads",
        #         TensorProto.INT64,
        #         [
        #             4,
        #         ],
        #         [
        #             0,
        #             0,
        #             1,
        #             1,
        #         ],
        #     )
        # ],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def)

    print("The graph in model:\n{}".format(model_def.graph))

    onnx.checker.check_model(model_def)

    # Save the model to a file
    onnx.save(model_def, "pad.onnx")


if __name__ == "__main__":
    main()
