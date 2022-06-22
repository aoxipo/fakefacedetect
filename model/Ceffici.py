from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from .efficientnet_pytorch.model import EfficientNet as Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()

class crop_model(nn.Module):
    def __init__(self, in_channel, num_classes) -> None:
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.line = torch.nn.Linear(18, 2)
        self.model = densenet(in_channel, num_classes, need_return_dic = False)
       

    def build_ans(self,x):
        return {
            "pred_logits":x,
        }

    def forward(self, input):
        try:
            p, c, _, _ = input[0].size()
        except Exception as e:
            print(e)
            print(len(input), input[0].size())
        # ans_ = torch.zeros(p,1)
        # if(use_gpu):
        #     ans_ = ans_.to(device)
        ans_ = []
        for i in input:
            ans = self.model(i)
            ans_.append(ans)
            # ans_ = torch.mul( ans_ ,ans)
        a = torch.stack( ans_, 1)

        #print(a.size())
        a = self.flatten(a)
        #print(a.size())
        ans_ = self.line(a)
        #print(ans_.size())
        return self.build_ans(ans_)

class crop_model1(nn.Module):
    def __init__(self, a,b, channel) -> None:
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.line = torch.nn.Linear(32, 2)
        self.model = Model(a,b,9)
        if(use_gpu):
            self.model.to(device)

    def build_ans(self,x):
        return {
            "pred_logits":x,
        }

    def forward(self, input):
        try:
            p, c, _, _ = input[0].size()
        except Exception as e:
            print(e)
            print(len(input), input[0].size())
        #method 1:
        # ans_ = torch.zeros(p,1)
        # if(use_gpu):
        #     ans_ = ans_.to(device)
        # ans_ = []
        # for i in input:
        #     ans = self.model(i)
        #     ans_.append(ans)
        #     # ans_ = torch.mul( ans_ ,ans)
        # a = torch.stack( ans_, 1)
        #method 2:
        input = self.crop_tensor(input)
        a = self.model(input)
        
        print(a.size())
        a = self.flatten(a)
        #print(a.size())
        ans_ = self.line(a)
        #print(ans_.size())
        return self.build_ans(ans_)

    def crop_tensor(self, image_pack, scale = 4):
        _, _, w, h = image_pack.size()
        a = int(w/scale)
        b = int(h/scale)
        t = torch.split(image_pack, a, dim = 2)
        ans = []
        for i in t:
            for j in torch.split(i,b, dim=3):
                ans.append(j)
        d = torch.cat(ans,1)
        return d