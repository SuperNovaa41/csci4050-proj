from model import TAModel
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("tensor_input")

args = parser.parse_args()

input_tens = list(map(int, args.tensor_input.split()))

model: TAModel = torch.load("saved_model.mdl", weights_only = False)

model.eval()


inp = torch.tensor(input_tens, dtype=torch.float32)
with torch.no_grad():
    x = inp.to("cpu")
    #_, preds = model(x)
    #preds = (preds > 0.5).float()
    #print(preds)


