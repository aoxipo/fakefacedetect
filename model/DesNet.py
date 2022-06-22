import torch
from torch import nn


# #__all__ = ['densenet121']
# model_urls = {
#     'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
#     'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
#     'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
#     'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
# }




def conv_block(in_channel, out_channel):
    layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),
        nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
    )
    return layer

class dense_block(nn.Module):
    def __init__(self, in_channel, growth_rate, num_layers):
        super(dense_block, self).__init__()
        block = []
        channel = in_channel
        for i in range(num_layers):
            block.append(conv_block(channel, growth_rate))
            channel += growth_rate
        self.net = nn.Sequential(*block)
    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat((out, x), dim=1)
        return x

def transition(in_channel, out_channel):
    trans_layer = nn.Sequential(
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),
        nn.Conv2d(in_channel, out_channel, 1),
        nn.AvgPool2d(2, 2)
    )
    return trans_layer

class densenet(nn.Module):
    def __init__(self, in_channel, num_classes, growth_rate=32, block_layers=[6, 12, 24, 16], need_return_dic = True):
        super(densenet, self).__init__()
        self.need_return_dict = need_return_dic
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, padding=1)
            )
        self.DB1 = self._make_dense_block(64, growth_rate,num=block_layers[0])
        self.TL1 = self._make_transition_layer(256)
        self.DB2 = self._make_dense_block(128, growth_rate, num=block_layers[1])
        self.TL2 = self._make_transition_layer(512)
        self.DB3 = self._make_dense_block(256, growth_rate, num=block_layers[2])
        self.TL3 = self._make_transition_layer(1024)
        self.DB4 = self._make_dense_block(512, growth_rate, num=block_layers[3])
        self.global_average = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.classifier = nn.Linear(1024, num_classes)
        #self.ea = nn.Linear(1024,2)
    def build_results(self,x):
        return {
            "pred_logits":x,
        }
    def forward(self, x):
        x = self.block1(x)
        x = self.DB1(x)
        x = self.TL1(x)
        x = self.DB2(x)
        x = self.TL2(x)
        x = self.DB3(x)
        x = self.TL3(x)
        x = self.DB4(x)
        x = self.global_average(x)
        x = x.view(x.shape[0], -1)
        #print(x.size())
        #a = self.ea(x)
        #print(a.size())
        x = self.classifier(x)
        #print(x.size())
        return self.build_results(x) if(self.need_return_dict) else x

    def _make_dense_block(self,channels, growth_rate, num):
        block = []
        block.append(dense_block(channels, growth_rate, num))
        channels += num * growth_rate

        return nn.Sequential(*block)
    def _make_transition_layer(self,channels):
        block = []
        block.append(transition(channels, channels // 2))
        return nn.Sequential(*block)
'''
def densenet121(pretrained=False, **kwargs):

    #model = dense_block(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)?
    if pretrained:      

        model_state = torch.load('/home/htu/manqi/PRA/PraNet-master/densenet121-a639ec97.pth',map_location='cpu')
        model.load_state_dict(model_state)
'''     
'''
if __name__ == '__main__':
    images = torch.rand(1, 3, 224, 224).cuda(0)
    model = densenet121(pretrained=True)
    model = model.cuda(0)
    print(model(images).size())
'''

# net = densenet(3,10)
# x = torch.rand(1,3,224,224)
# for name,layer in net.named_children():
#     if name != "classifier":
#         x = layer(x)
#         print(name, 'output shape:', x.shape)
#     else:
#         x = x.view(x.size(0), -1)
#         x = layer(x)
#         print(name, 'output shape:', x.shape)

class crop_model_cat(nn.Module):
    def __init__(self, in_channel, num_classes) -> None:
        super().__init__()
        middle_layer_number = 9*2*10
        self.flatten = torch.nn.Flatten()
        self.line = torch.nn.Linear(middle_layer_number, 2)
        self.model = densenet(1, num_classes*32, need_return_dic = False)

        # self.attention = torch.nn.Sequential(
        #     torch.nn.Linear(middle_layer_number,num_classes),
        #     torch.nn.ReLU(),
        # )
        # self.BN = torch.nn.Sequential(
        #     torch.nn.BatchNorm1d(middle_layer_number)
        # )
        # self.dense = torch.nn.Sequential( 
        #     torch.nn.Linear(middle_layer_number,64),
        #     torch.nn.ReLU(),
        # )
        # self.dense2 = torch.nn.Sequential( 
        #     torch.nn.Linear(64,32),
        #     torch.nn.ReLU(),
        # )
        # self.dense3 = torch.nn.Sequential( 
        #     torch.nn.Linear(32, num_classes),
        #     torch.nn.ReLU(),
        # )

   
    # def feature(self, x):
    #     x = self.Flatten(x)
    #     x = self.BN(x)
    #     p = self.attention(x)
    #     x = self.dense(x)
        
    #     x = self.dense2(x)
        
    #     x = self.dense3(x)
        
    #     x =  torch.mul(x,p)
    #     return x

    def build_ans(self,x):
        return {
            "pred_logits":x,
        }

    def revert_tensor(self,image_pack, batch):
        ans = []
        for i in range(batch):
            ans.append(image_pack[i::batch,:])
        return torch.stack(ans)

    def forward(self, input):
        # try:
        #     p, c, _, _ = input[0].size()
        # except Exception as e:
        #     print(e)
        #     print(len(input), input[0].size())
        #print(input.size())
        p, c, _, _ = input.size()
        input = self.crop_tensor(input,3)
        #print(input.size())
        a = self.model(input)
        
        #print(a.size())
        # revert
        
        a = self.revert_tensor(a, p)
        print(a.size())
        a = self.flatten(a)
        #a = self.feature(a)

        #return self.build_ans(a)
        # #print(a.size()," flatten \n")
        # #print(a.size())
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
        d = torch.cat(ans,0)
        return d

class crop_model(nn.Module):
    def __init__(self, in_channel, num_classes) -> None:
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.line = torch.nn.Linear(90, 2)
        self.model = densenet(in_channel, 10, need_return_dic = False)
       

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
