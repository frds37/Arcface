import numbers
import mxnet as mx
import torch
import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import numpy as np

class get_data(Dataset):
    def __init__(self):
        loc_data = '/home/cryptology96/FaceRecognition/Challenge1/DS/dataset'
        loc_imgrec = os.path.join(loc_data, 'train.rec')
        loc_imgidx = os.path.join(loc_data, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(loc_imgidx, loc_imgrec, 'r')
        item = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(item)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.idx = np.array(range(1, int(header.label[0])))            
        else:
            self.idx = np.array(list(self.imgrec.keys))
            
        self.transform = T.Compose(
            [T.ToPILImage(),
             T.RandomHorizontalFlip(),
             T.RandomVerticalFlip(),
             T.ToTensor(),
             T.Resize((224,224)),
             T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        
    def __getitem__(self, index):
        ind = self.idx[index]
        item = self.imgrec.read_idx(ind)
        header, img = mx.recordio.unpack(item)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label).long()
        image = mx.image.imdecode(img).asnumpy()
        image = self.transform(image)
        
        return image, label

    def __len__(self):
        return len(self.idx)
       