import torch
import argparse


if torch.cuda.is_available():
    print('Saving JIT model from GPU device')
    print(torch.__version__)

    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model_path", help="the location of the model")
    args = argParser.parse_args()

    model = torch.jit.load(args.model_path, map_location='cuda')
    model = model.to('cuda')

    model_input=(torch.tensor([[10,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0]]).to('cuda'), torch.tensor([1]).to('cuda'))
    print(model.forward(model_input[0], model_input[1]))
    torch.jit.save(model, args.model_path)
else:
    print('No GPU found, thus conversion of model is not possible')

