# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
from data.dataset import get_data_transforms,MedicalDataset
import numpy as np
from model.dynamic_tanh import convert_ln_to_dyt
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from model.resnet import generate_model, FeatureFusionClassifier
from test import evaluation
from torch.nn import functional as F

from model.vit3d_gate import ViT3D

def cross_entropy_loss(outputs, target_onehot):
    loss1 = -torch.sum(target_onehot * F.log_softmax(outputs, dim=1))
    return loss1


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train():
    #ad vs cn task the epoch is 100
    #lmci vs emci the epoch is 120
    epochs = 100

    batch_size = 2
    image_size = 96
        
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)

    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    train_path = "/data/birth/cyf/shared_data/CAPL/new/new_register/mri"
    test_path = "/data/birth/cyf/shared_data/CAPL/new/new_register/mri"
    ptrain_path = "/data/birth/cyf/shared_data/CAPL/new/new_register/pet"
    ptest_path = "/data/birth/cyf/shared_data/CAPL/new/new_register/pet"
    ckp_path = '/data/birth/cyf/output/CAPL/mymodel/task2/last_model.pth'
    ckp_path_best = '/data/birth/cyf/output/CAPL/mymodel/task2/best_model.pth'
    #train_data = MedicalDataset(root=train_path, transform=None)
    train_data = MedicalDataset(root=train_path, rootp=ptrain_path,transform=data_transform, phase="train")
    test_data = MedicalDataset(root=test_path, rootp=ptest_path,transform=data_transform, phase="test")


    # train_sample = train_data[0]
    # val_sample = val_data[0]
    # print("Train label type:", type(train_sample[2]), "value:", train_sample[2])
    # print("Val label type:", type(val_sample[2]), "value:", val_sample[2])

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    en1 = generate_model(50).to(device)
    en2 = generate_model(50).to(device)
    fuse1=FeatureFusionClassifier(en1).to(device)
    fuse2 = FeatureFusionClassifier(en2).to(device)
    decoder = ViT3D(
        image_size=(12, 12, 12),
        patch_size=12,
        num_classes=2,
        dim=1024,  # token dim
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.3,
        emb_dropout=0.2,
        num_batches=2048
    )
    decoder = convert_ln_to_dyt(decoder)


    # optimizer = torch.optim.Adam(list(decoder.parameters())+list(bn1.parameters()), lr=learning_rate, betas=(0.5,0.999))
    #model_parameters1 = list(en1.parameters()) + list(fuse1.parameters())
    model_parameters1 = list(en1.parameters())
    #model_parameters2 = list(en2.parameters()) + list(fuse2.parameters())
    model_parameters2 = list(en2.parameters())
    model_parameters3 = list(decoder.parameters())
    #编码器1
    optimizer1=torch.optim.Adam(model_parameters1, lr=1e-5, weight_decay=1e-4)
    #scheduler1 = PolyLR(optimizer1, max_epoch=int(epochs), power=0.9)
    optimizer2=torch.optim.Adam(model_parameters2, lr=1e-5, weight_decay=1e-4)
    #scheduler2 = PolyLR(optimizer2, max_epoch=int(epochs), power=0.9)
    optimizer3 = torch.optim.Adam(model_parameters3, lr=5e-5, betas=(0.9, 0.999), weight_decay=1e-4,amsgrad=True)
    #scheduler3 = PolyLR(optimizer3, max_epoch=int(epochs), power=0.9)

    best_val_acc = float('-inf')  # 跟踪验证集最佳准确率
    best_acc = float('-inf')  # 跟踪测试集最佳准确率（仅在验证集表现好时更新）

    for epoch in range(epochs):
        en1.train()
        en2.train()
        fuse1.train()
        fuse2.train()
        decoder.train()
        loss_list1 = []
        loss_list2 = []
        loss_list3 = []
        min_loss=float('inf')
        # if epoch==5:
        #     break
        for i,(img , pet , label,_) in enumerate(train_dataloader):
            #print(f"Sample {i}: Label = {label}, Input shape = {img.shape}, Input range = [{img.min()}, {img.max()}]")
            # if i == 1:
            #     print("test over")
            #     break
            print("img:",img.shape)
            # print("pet:",pet.shape)
            img ,pet, label = img.to(device), pet.to(device),label.to(device)
            batch_size=img.size(0)
            en1,en2,fuse1,fuse2,decoder=en1.to(device),en2.to(device),fuse1.to(device),fuse2.to(device),decoder.to(device)

            # 准备target（放在循环内部）
            target1 = torch.tensor([0]).to(device)
            target_onehot1 = F.one_hot(target1, num_classes=2).float().to(device)
            target_onehot1 = target_onehot1.repeat(batch_size, 1)

            target2 = torch.tensor([1]).to(device)
            target_onehot2 = F.one_hot(target2, num_classes=2).float().to(device)
            target_onehot2 = target_onehot2.repeat(batch_size, 1)

            target_onehot3 = F.one_hot(label, num_classes=2).float().to(device)
            #print("3:",target_onehot3.shape,target_onehot3)



            target = torch.tensor([0.5, 0.5]).to(device)
            target = target.repeat(batch_size, 1)

            feature1, output1 = en1(img)  # en1的前向传播
            feature2, output2 = en2(pet)  # en2的前向传播

            # 1. 计算loss1（更新en1和fuse1）
            optimizer1.zero_grad()
            fuse_feature1, fuseoutput1 = fuse1(feature1, feature2.detach())  # 隔离en2的梯度
            loss1 = 0.5 * cross_entropy_loss(output1, target_onehot1) + 0.5 * cross_entropy_loss(fuseoutput1, target)
            #print('loss1',loss1.item())
            loss_list1.append(loss1.item())
            loss1.backward()
            optimizer1.step()

            # 2. 计算loss2（更新en2和fuse2）
            optimizer2.zero_grad()
            fuse_feature2, fuseoutput2 = fuse2(feature2, feature1.detach())  # 隔离en1的梯度
            loss2 = 0.5 * cross_entropy_loss(output2, target_onehot2) + 0.5 * cross_entropy_loss(fuseoutput2, target)
            #print('loss2',loss2.item())
            loss_list2.append(loss2.item())
            loss2.backward()
            optimizer2.step()

            # 3. 计算loss3（更新decoder）
            optimizer3.zero_grad()
            output = decoder((fuse_feature1.detach() + fuse_feature2.detach())/2)
            #print("output:",output)
            loss3 = cross_entropy_loss(output, target_onehot3)
            loss_list3.append(loss3.item())
            loss3.backward()
            optimizer3.step()

        print('epoch [{}/{}], loss1:{:.4f},loss2:{:.4f},loss3:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list1),np.mean(loss_list2),np.mean(loss_list3)))

        if (epoch + 1) % 5 == 0:
            acc, f1, kappa,sen ,precis, mcc = evaluation(en1, en2,fuse1,fuse2,decoder, test_dataloader, device)
            print('Validation - acc:{:.4f}, f1{:.4f}, kappa{:.4f}, sen{:.4f}, precision{:.4f}, mcc{:.4f}'.format(acc, f1, kappa, sen, precis, mcc))
            if acc > best_val_acc:
                best_val_acc = acc
                torch.save({'en1': en1.state_dict(), 'en2': en2.state_dict(), 'fuse1:': fuse1.state_dict(),
                            'fuse2:': fuse2.state_dict(),
                            'decoder': decoder.state_dict(), "optim1:": optimizer1.state_dict(),
                            "optim2:": optimizer2.state_dict(), "optim3:": optimizer3.state_dict()}, ckp_path_best)
                print("best_val_acc:", best_val_acc, "epoch:", epoch)
            torch.save({'en1': en1.state_dict(),'en2': en2.state_dict(),'fuse1:':fuse1.state_dict(),'fuse2:':fuse2.state_dict(),
                        'decoder': decoder.state_dict(),"optim1:":optimizer1.state_dict(),"optim2:":optimizer2.state_dict(),"optim3:":optimizer3.state_dict()}, ckp_path)
    #最终测试
    print('Finished Training')
    checkpoint = torch.load(ckp_path_best)
    en1.load_state_dict(checkpoint['en1'])
    en2.load_state_dict(checkpoint['en2'])
    fuse1.load_state_dict(checkpoint['fuse1:'])
    fuse2.load_state_dict(checkpoint['fuse2:'])
    decoder.load_state_dict(checkpoint['decoder'])
    print('----Testing----')
    acc, f1, kappa, sen, precis, mcc = evaluation(en1, en2, fuse1, fuse2, decoder, test_dataloader, device)
    print('Validation - acc:{:.4f}, f1{:.4f}, kappa{:.4f}, sen{:.4f}, precision{:.4f}, mcc{:.4f}'.format(acc, f1, kappa, sen, precis, mcc))
    return acc, f1, kappa




if __name__ == '__main__':

    setup_seed(1037)
    train()

