import os

import torch

def export_model_to_onnx(model, bin_op, output_model_name):
    sample_input = torch.rand((1, 3, 224, 224))
    output_path = os.path.join('checkpoints', output_model_name)

    # Export the model
    model.eval()
    model.to(torch.device('cpu'))
    bin_op.binarization()
    torch.onnx.export(model,                                    # model being run
                        sample_input,                           # model input (or a tuple for multiple inputs)
                        output_path,                            # where to save the model (can be a file or file-like object)
                        export_params=True,                     # store the trained parameter weights inside the model file
                        # do_constant_folding=True,             # whether to execute constant folding for optimization
                        input_names = ['input'],                # the model's input names
                        output_names = ['output'],              # the model's output names
                        # keep_initializers_as_inputs=True      # Needed if you want to convert opset version
                    )
    bin_op.restore()
    model.train()
