import torch
from torch.autograd import Variable
import numpy as np
from train import params
from util import utils
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

def train(training_mode,
          feature_extractor, class_classifier,domain_classifier,
          class_criterion, domain_criterion,
          source_dataloader, target_dataloader, optimizer, epoch):
    """
    Execute target domain adaptation
    :param training_mode:训练模式
    :param feature_extractor: 特征提取器网络
    :param class_classifier:类别分类器网络
    :param domain_classifier:域分类器网络
    :param class_criterion: 类别分类损失函数
    :param domain_criterion:域分类损失函数
    :param source_dataloader:源域数据加载器
    :param target_dataloader:目标域数据加载器
    :param optimizer:优化器
    :return:
    """

    total_class_loss = 0
    total_domain_loss = 0
    total_loss = 0
    num_batches = 0

    # 特征提取器，分类器和领域分类器设置为训练模式，需要更新梯度
    feature_extractor.train()
    class_classifier.train()
    domain_classifier.train()

    # steps当前epoch之前的所有训练步数
    start_steps = epoch * len(source_dataloader)
    total_steps = params.epochs * len(source_dataloader)

    epoch_loss = []
    # source和target数据打包成组
    for batch_idx, (sdata, tdata) in enumerate(zip(source_dataloader, target_dataloader)):
        num_batches += 1
        if training_mode == 'dann':
            # p: 当前训练进度的比例，用于调整梯度反转层的参数
            # constant: 梯度反转层的缩放参数
            p = float(batch_idx + start_steps) / total_steps
            constant = 2. / (1. + np.exp(-params.gamma * p)) - 1

            # 准备数据，切片，移动到GPU上
            input1, label1 = sdata
            input2, label2 = tdata
            size = min((input1.shape[0], input2.shape[0]))
            input1, label1 = input1[0:size, :, :, :], label1[0:size]
            input2, label2 = input2[0:size, :, :, :], label2[0:size]
            if params.use_gpu:
                input1, label1 = Variable(input1.cuda()), Variable(label1.cuda())
                input2, label2 = Variable(input2.cuda()), Variable(label2.cuda())
            else:
                input1, label1 = Variable(input1), Variable(label1)
                input2, label2 = Variable(input2), Variable(label2)

            # 根据当前进度更新优化器的学习率，并清零梯度。
            optimizer = utils.optimizer_scheduler(optimizer, p)
            optimizer.zero_grad()

            # source_labels: 源域标签，全部为0。target_labels: 目标域标签，全部为1。
            if params.use_gpu:
                source_labels = Variable(torch.zeros((input1.size()[0])).type(torch.LongTensor).cuda())
                target_labels = Variable(torch.ones((input2.size()[0])).type(torch.LongTensor).cuda())
            else:
                source_labels = Variable(torch.zeros((input1.size()[0])).type(torch.LongTensor))
                target_labels = Variable(torch.ones((input2.size()[0])).type(torch.LongTensor))

            # 计算得到2个样本th的特征向量
            src_feature = feature_extractor(input1)
            tgt_feature = feature_extractor(input2)

            # 通过类别分类器计算源域的类别预测，并计算其类别损失。
            class_preds = class_classifier(src_feature)
            class_loss = class_criterion(class_preds, label1)

            # compute the domain loss of src_feature and target_feature
            # 通过域分类器计算源域和目标域的域预测，并计算其域损失。
            tgt_preds = domain_classifier(tgt_feature, constant)
            src_preds = domain_classifier(src_feature, constant)
            tgt_loss = domain_criterion(tgt_preds, target_labels)
            src_loss = domain_criterion(src_preds, source_labels)
            domain_loss = tgt_loss + src_loss

            # 总损失由类别损失和域损失加权和构成。
            theta = 2. / (1. + np.exp(-params.gamma * p)) - 1
            #theta = params.theta
            loss = class_loss + theta * domain_loss

            total_class_loss += class_loss.item()
            total_domain_loss += domain_loss.item()
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            # print loss
            if (batch_idx + 1) % 10 == 0:
                print('[{}/{} ({:.0f}%)]\tLoss: {:.6f}\tClass Loss: {:.6f}\tDomain Loss: {:.6f}'.format(
                    batch_idx * len(input2), len(target_dataloader.dataset),
                    100. * batch_idx / len(target_dataloader), loss.item(), class_loss.item(),
                    domain_loss.item()
                ))


        # 源域训练模式
        elif training_mode == 'source':
            # prepare the data
            input1, label1 = sdata
            # .shape[0],批次大小
            size = input1.shape[0]
            # input是 B,C，H,M。只选择[0,SIZE)个样本，label同理
            input1, label1 = input1[0:size, :, :, :], label1[0:size]

            # 数据移到GPU上
            if params.use_gpu:
                input1, label1 = Variable(input1.cuda()), Variable(label1.cuda())
            else:
                input1, label1 = Variable(input1), Variable(label1)

            # setup SGD optimizer,特征提取和标签分类一起更新
            optimizer = optim.SGD(list(feature_extractor.parameters())+list(class_classifier.parameters()), lr=0.01, momentum=0.9)
            optimizer.zero_grad()

            # 特征提取器对输入数据 input1 进行前向传播，得到源域的特征表示 src_feature
            src_feature = feature_extractor(input1)

            # 类别分类器对特征 src_feature 进行前向传播，得到类别预测值 class_preds
            class_preds = class_classifier(src_feature)
            class_loss = class_criterion(class_preds, label1)

            # 计算梯度，反向传播，使类别区分更准
            class_loss.backward()
            optimizer.step()

            total_class_loss += class_loss.item()
            total_loss += class_loss.item()

            # print loss
            if (batch_idx + 1) % 10 == 0:
                print('[{}/{} ({:.0f}%)]\tClass Loss: {:.6f}'.format(
                    batch_idx * len(input1), len(source_dataloader.dataset),
                    100. * batch_idx / len(source_dataloader), class_loss.item()
                ))


        elif training_mode == 'target':
            # prepare the data
            input2, label2 = tdata
            size = input2.shape[0]
            input2, label2 = input2[0:size, :, :, :], label2[0:size]
            if params.use_gpu:
                input2, label2 = Variable(input2.cuda()), Variable(label2.cuda())
            else:
                input2, label2 = Variable(input2), Variable(label2)

            # setup optimizer
            optimizer = optim.SGD(list(feature_extractor.parameters()) + list(class_classifier.parameters()), lr=0.01,
                                  momentum=0.9)
            optimizer.zero_grad()
            # 计算特征
            tgt_feature = feature_extractor(input2)

            # 计算特征的类别损失
            class_preds = class_classifier(tgt_feature)
            class_loss = class_criterion(class_preds, label2)

            # 最小化类别损失
            class_loss.backward()
            optimizer.step()

            total_class_loss += class_loss.item()
            total_loss += class_loss.item()

            # print loss
            if (batch_idx + 1) % 10 == 0:
                print('[{}/{} ({:.0f}%)]\tClass Loss: {:.6f}'.format(
                    batch_idx * len(input2), len(target_dataloader.dataset),
                    100. * batch_idx / len(target_dataloader), class_loss.item()
                ))


    average_class_loss = total_class_loss / num_batches
    average_domain_loss = total_domain_loss / num_batches if training_mode == 'dann' else 0
    average_total_loss = total_loss / num_batches

    return average_total_loss,average_class_loss,average_domain_loss
