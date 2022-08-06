# https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
import torch
import torchvision
import torch.nn as nn
import onnx
import onnxruntime
import numpy as np

#model = torchvision.models.mobilenet_v3_small(pretrained=True)
model_name = 'mobilenet_v2'

if model_name == 'mobilenet_v3':
    model = torchvision.models.mobilenet_v3_large()

    last_channel = 1280
    lastconv_output_channels = 960
    num_classes = 3

    model.classifier[3] = nn.Linear(last_channel,num_classes)

    model.load_state_dict(torch.load("../models/2022-06-28/mobilenet_v3_large_finetuned.pt"))
    model.eval()
    #x = torch.zeros(1, 3, 244, 244)
    #print(model(x))

    save_path = "../models/2022-06-28/mobilenet_v3_large_finetuned.onnx"

else:
    model = torchvision.models.mobilenet_v2()

    last_channel = 1280
    num_classes = 3

    model.classifier[1] = nn.Linear(last_channel,num_classes)

    model.load_state_dict(torch.load("../models/2022-07-23/mobilenet_v2_finetuned.pt"))
    model.eval()
    x = torch.zeros(1, 3, 244, 244)
    print(model(x))

    save_path = "../models/2022-07-23/mobilenet_v2_finetuned.onnx"    
'''
batch_size = 1

x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
torch_out = model(x)

torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  save_path,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=9,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})


onnx_model = onnx.load(save_path)
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession(save_path)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")

'''