import torch
import easydict
import Loss
import Backbone
import torch.utils.data
import torch.nn.functional as F
import call_log
import data
from verify_call import CallBackVerification, CallBackLogging
from setting import setting as setn


def main(args):
#logging 세팅    
    call_log.make_logger()
    torch.autograd.set_detect_anomaly(True)
#GPU setting
    device = 'cuda:' + str(args.GPU_num) 
    torch.cuda.set_device(device)
    
# 이미지 받아오기        
    train_set = data.get_data()
    train_sampler = torch.utils.data.RandomSampler(train_set)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, sampler=train_sampler, batch_size=setn.batch_size)
    
#loss function과 network 선택
    loss_f = Loss.get_loss(args.loss)        
    backbone = Backbone.call_net(args.network)
    backbone.train()
    
#optimizer 설정
    opt_backbone = torch.optim.SGD([{'params': backbone.parameters()}], lr=setn.lr, momentum=setn.momentum, weight_decay=setn.weight_decay)
    opt_loss = torch.optim.SGD([{'params': loss_f.parameters()}], lr=setn.lr, momentum=setn.momentum, weight_decay=setn.weight_decay)
    
#leraning rate 설정    
    lr_lambd = lambda epoch : 0.1 ** (1+(epoch>20)+(epoch>28)+(epoch>32))
    blr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt_backbone, lr_lambda=lr_lambd)
    llr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt_loss, lr_lambda=lr_lambd)
    
#verification 세팅
    total_step = int(len(train_set) / setn.batch_size * setn.epoch_num)
    CallVerification = CallBackVerification(1000, setn.target, setn.folder)
    CallLogging = CallBackLogging(40, total_step, setn.batch_size, None)
    
# 학습과정  

# 1. 받아온 이미지를 backbone에 넣기
# 2. backbone에서 나온 feature를 header에 넣어서 output으로 나온 vector를 loss function에 넣어 loss 값 계산
# 3. 계산한 loss를 바탕으로 back propagation 실시
# 4. parameter, lr scheduler update
    
    step = 0
    for epoch in range(setn.epoch_num):
        for (img, label) in train_loader:
            step += 1
            print(step)
            feature = F.normalize(backbone(img))
            loss = loss_f(feature, label)
            loss.backward()
            opt_backbone.step()
            opt_loss.step()
            opt_backbone.zero_grad()
            opt_loss.zero_grad()
            CallVerification(step, backbone)
            CallLogging(step, loss, epoch)
            
        torch.save(backbone.module.state_dict(), 'backbone.pth')
        logging.info("Model Saved Sucessfully in '{}'".format('backbone.pth'))
        
        blr_scheduler.step()
        llr_scheduler.step()
        
    
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args_ = easydict.EasyDict({
        "GPU_num" : 1,
        "network" : 'r50',
        "loss" : 'Arcface'
    })
    main(args_)
