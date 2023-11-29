from dataset import MyDataset
from net import SimpleConv3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import matplotlib.pyplot as plt



traindata_transformer =transforms.Compose([transforms.Resize(60),
                                           transforms.RandomCrop(48),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomRotation(10),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                           ])
valdata_transformer =transforms.Compose([
                                           transforms.Resize(48),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                            ])
if __name__ == '__main__':
    traintxt='data/train.txt'
    valtxt='data/val.txt'
    traindataset=MyDataset(traintxt,traindata_transformer)
    valdataset = MyDataset(valtxt, valdata_transformer)
    print('训练数据总数'+str(traindataset.__len__()))
    print('验证数据总数'+str(valdataset.__len__()))

    net = SimpleConv3(2)
    traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=64, shuffle=True, num_workers=1)
    valdataloader = torch.utils.data.DataLoader(valdataset, batch_size=64, shuffle=False, num_workers=1)

    optim = SGD(net.parameters(),lr=0.05, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    lr_step = StepLR(optim, step_size=50,gamma=0.1)
    epochs = 1


    accs=[]
    losss=[]

    for epoch in range(0,epochs):
        batch=0
        lr_step.step()
        running_acc = 0.0
        running_loss= 0.0
        for data in traindataloader:
            batch += 1
            input , target = data
            output = net(input)
            loss = criterion(output,target)

            acc = float(torch.sum(target == torch.argmax(output,1))) /len(input)
            running_acc += acc
            running_loss += loss.data.item()
            optim.zero_grad()
            loss.backward()
            optim.step()

            running_acc = running_acc/batch
            running_loss = running_loss/batch
            accs.append(running_acc)
            losss.append(running_loss)

            print('epoch='+str(epoch)+' loss='+str(running_loss)+' acc='+str(running_acc))

        torch.save(net,'model.pth')
        x = torch.randn(1,3,48,48)

        net = torch.load('model.pth')
        net.train(False)
        torch.onnx.export(net,x,'model.onnx')


        plt.plot(range(len(accs)),accs)
        plt.plot(range(len(losss)), losss)
        plt.xlabel('epoch')
        plt.legend(('acc','loss'))

        plt.show()
