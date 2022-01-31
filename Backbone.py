import torch
import torch.nn as nn
from setting import setting as setn


def conv1x1(in_channels, out_channels, stride, padding):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding, bias=False)        
        
def conv3x3(in_channels, out_channels, stride, padding):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False)

def conv7x7(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        

class init_block(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.layer1 = conv7x7(3, 64)
        self.norm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pad = nn.ZeroPad2d(1)
        self.maxpool = nn.MaxPool2d(3, stride=2)
        
    def forward(self, x):
        output = self.layer1(x)
        output = self.norm(output)
        output = self.relu(output)
        output = self.pad(output)
        output = self.maxpool(output)
        
        return output
    
class main_block(nn.Module):
    def __init__(self, num_feature, stride, use1, count):
        super().__init__()
        self.use1 = use1
        self.count = count
        self.num_feature = num_feature
        self.oconv1_1 = conv3x3(int(num_feature * 2), num_feature, 1, 'same')
        self.oconv1_2 = conv3x3(num_feature, num_feature, 1, 'same')        
        self.oconv2 = conv3x3(num_feature, num_feature, 1, 'same')
        self.conv1_1 = conv1x1(int(num_feature * 2), num_feature, stride, 'valid')
        self.conv1_2 = conv1x1(num_feature, num_feature, stride, 'valid')
        self.conv1_3 = conv1x1(num_feature * 4, num_feature, 1, 'valid')
        self.conv2 = conv3x3(num_feature, num_feature, 1, 'same')
        self.conv3 = conv1x1(num_feature, num_feature * 4, 1, 'valid')
        self.conv3_2 = conv1x1(num_feature * 2, num_feature * 4, 2, 'valid')
        self.norm1 = nn.BatchNorm2d(num_feature)
        self.norm2 = nn.BatchNorm2d(num_feature * 4)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        if self.use1 == False:
            temp = x
            if self.count == 0 and self.num_feature != 64:
                x = self.oconv1_1(x)            
            else:
                 x = self.oconv1_2(x)
            x = self.norm1(x)
            x = self.relu(x)
            x = self.oconv2(x)
            x = self.norm1(x)
            x += temp
            output = self.relu(x)
        else:
            temp = x
            if self.count == 0 and self.num_feature != 64:
                x = self.conv1_1(x)
            elif self.count == 0 and self.num_feature == 64:
                x = self.conv1_2(x)
            else:
                x = self.conv1_3(x)
            x = self.norm1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.norm1(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.norm2(x)
            if self.count == 0:
                if self.num_feature == 64:
                    temp = self.conv3(temp)
                else:
                    temp = self.conv3_2(temp)
                temp = self.norm2(temp) 
            x = x + temp
            output = self.relu(x)
            
        return output

            
class conv_block(nn.Module):
    
    def __init__(self, layers, use1):
        super().__init__()
        self.layers = layers
        self.use1 = use1
        self.relu = nn.ReLU()
        self.layer1 = self.block(64, self.layers[0], self.use1, 1)
        self.layer2 = self.block(128, self.layers[1], self.use1, 2)
        self.layer3 = self.block(256, self.layers[2], self.use1, 2)
        self.layer4 = self.block(512, self.layers[3], self.use1, 2)        
        
        
    def block(self, num_feature, layer_num, use_1, stride):
        layer = []
        for count in range(layer_num):
            layer.append(main_block(num_feature, stride, use_1, count))        
                
        return nn.Sequential(*layer)
        
    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.relu(output)
        
        return output
        
class blockbuilding(nn.Module):
    
    def __init__(self, layers, use1):
        super().__init__()
        self.layers = layers
        self.use1 = use1
        self.pool = nn.AvgPool2d(7)
        self.fc = nn.Linear(2048, setn.embedding_size)
        
#         for i in self.modules():
#             if isinstance(i, nn.Conv2d):
#                 nn.init.normal_(i.weight, 0, 0.1)
#             elif isinstance(i, nn.BatchNorm2d):
#                 nn.init.constant_(i.weight, 1)
#                 nn.init.constant_(i.bias, 0)
        
    def forward(self, x):
#         with torch.cuda.amp.autocast():
        output = init_block()(x)
        output = conv_block(self.layers, self.use1)(output)
        output = self.pool(output)
        output = output.view(x.size(0), 1, -1)
        output = self.fc(output)

        return output
        
        
def call_net(name):
    if name == "r18":
        return blockbuilding([2, 2, 2, 2], False)
    elif name == "r34":
        return blockbuilding([3, 4, 6, 3], False)
    elif name == "r50":
        return blockbuilding([3, 4, 6, 3], True)
    elif name == "r101":
        return blockbuilding([3, 4, 23, 3], True)
        
        
        
        
        
        
        
        
        
        