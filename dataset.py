import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

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
class MyDataset(torch.utils.data.Dataset):
    def __init__(self,datatxt,datatransform):
        datas = open(datatxt,'r').readlines()
        self.images = []
        self.labels = []
        self.transform = datatransform
        for data in datas:
            item = data.strip().split(' ')
            self.images.append(item[0])
            self.labels.append(item[1])
        print('number of datas ='+str(len(self.images)))
        return

    def __getitem__(self, item):
        imagepath,label = self.images[item],self.labels[item]
        image = Image.open(imagepath)
        return self.transform(image),int(label)

    def __len__(self):
        return len(self.images)



    # traindataset = datasets.ImageFolder(datatrain_dir,traindata_transformer)
    # valdataset=datasets=datasets.ImageFolder(dataval_dir,valdata_transformer)

if __name__ == '__main__':
    traintxt='data/train.txt'
    valtxt='data/val.txt'
    traindataset=MyDataset(traintxt,traindata_transformer)
    valdataset = MyDataset(valtxt, valdata_transformer)
    print('训练数据总数'+str(traindataset.__len__()))
    print('验证数据总数'+str(valdataset.__len__()))

    # datatrain_dir = '../data/train'
    # dataval_dir = '../data/val'
    # traindataloader = torch.utils.data.DataLoader(traindataset, batch_size=16, shuffle=True, num_workers=1)
    # valdataloader = torch.utils.data.DataLoader(valdataset, batch_size=16, shuffle=True, num_workers=1)

    traindataloader = torch.utils.data.DataLoader(traindataset,batch_size=16,shuffle=True,num_workers=1)
    valdataloader = torch.utils.data.DataLoader(valdataset,batch_size=16,shuffle=False,num_workers=1)

    # 遍历训练数据集的一个 minibatch
    for sample in traindataloader:
        # print(type(sample))
        print(type(sample[0]))
        print(sample[0].shape)
        print(sample[0])
        print(type(sample[1]))
        print(sample[1].shape)
        print(sample[1])

