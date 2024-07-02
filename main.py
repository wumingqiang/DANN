"""
Main script for models
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import torch.nn as nn
import torch.optim as optim
import matplotlib as plt
import numpy as np

from models import models
from train import test, train, params
from util import utils
from sklearn.manifold import TSNE
import argparse, sys, os

import torch
from torch.autograd import Variable
import time


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


def visualizePerformance(feature_extractor, class_classifier, domain_classifier, src_test_dataloader,
                         tgt_test_dataloader, num_of_samples=None, imgName=None):
    """
    Evaluate the performance of dann and source only by visualization.
    将高维特征嵌入到二维空间中，并通过图像展示源域和目标域样本的分布情况
    :param feature_extractor: network used to extract feature from target samples
    :param class_classifier: network used to predict labels
    :param domain_classifier: network used to predict domain
    :param source_dataloader: test dataloader of source domain
    :param target_dataloader: test dataloader of target domain
    :param num_of_samples: the number of samples (from train and test respectively) for t-sne
    :param imgName: the name of saving image

    :return:
    """

    # 将网络设置为评估模式
    feature_extractor.eval()
    class_classifier.eval()
    domain_classifier.eval()

    # Randomly select samples from source domain and target domain.
    # 随机选取一部分数据
    if num_of_samples is None:
        num_of_samples = params.batch_size
    else:
        assert len(src_test_dataloader) * num_of_samples, \
            'The number of samples can not bigger than dataset.' # NOT PRECISELY COMPUTATION

    # Collect source data.
    s_images, s_labels, s_tags = [], [], []
    for batch in src_test_dataloader:
        images, labels = batch

        if params.use_gpu:
            s_images.append(images.cuda())
        else:
            s_images.append(images)
        s_labels.append(labels)
        s_tags.append(torch.zeros((labels.size()[0])).type(torch.LongTensor))

        if len(s_images * params.batch_size) > num_of_samples:
            break

    s_images, s_labels, s_tags = (torch.cat(s_images)[:num_of_samples],
                                 torch.cat(s_labels)[:num_of_samples],
                                 torch.cat(s_tags)[:num_of_samples])


    # Collect test data.
    t_images, t_labels, t_tags = [], [], []
    for batch in tgt_test_dataloader:
        images, labels = batch

        if params.use_gpu:
            t_images.append(images.cuda())
        else:
            t_images.append(images)
        t_labels.append(labels)

        t_tags.append(torch.ones((labels.size()[0])).type(torch.LongTensor))

        if len(t_images * params.batch_size) > num_of_samples:
            break

    t_images, t_labels, t_tags = (torch.cat(t_images)[:num_of_samples],
                                 torch.cat(t_labels)[:num_of_samples],
                                 torch.cat(t_tags)[:num_of_samples])

    # Compute the embedding of target domain.
    embedding1 = feature_extractor(s_images)
    embedding2 = feature_extractor(t_images)

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)

    if params.use_gpu:
        dann_tsne = tsne.fit_transform(np.concatenate((embedding1.cpu().detach().numpy(),
                                                       embedding2.cpu().detach().numpy())))
    else:
        dann_tsne = tsne.fit_transform(np.concatenate((embedding1.detach().numpy(),
                                                   embedding2.detach().numpy())))

    utils.plot_embedding(dann_tsne, np.concatenate((s_labels, t_labels)),
                         np.concatenate((s_tags, t_tags)), 'Domain Adaptation', imgName)



def main(args):

    # 设置全局参数
    params.fig_mode = args.fig_mode
    params.epochs = args.max_epoch
    params.training_mode = args.training_mode
    params.source_domain = args.source_domain
    params.target_domain = args.target_domain
    if params.embed_plot_epoch is None:
        params.embed_plot_epoch = args.embed_plot_epoch
    params.lr = args.lr

    # 如果指定了保存路径，将其保存到全局参数中，否则默认将图片保存在./experiment文件夹中。
    if args.save_dir is not None:
        params.save_dir = args.save_dir
    else:
        print('Figures will be saved in ./experiment folder.')

    # prepare the source data and target data

    src_train_dataloader = utils.get_train_loader(params.source_domain)
    src_test_dataloader = utils.get_test_loader(params.source_domain)
    tgt_train_dataloader = utils.get_train_loader(params.target_domain)
    tgt_test_dataloader = utils.get_test_loader(params.target_domain)

    # if params.fig_mode is not None:
    #     print('Images from training on source domain:')
    #     utils.displayImages(src_train_dataloader, imgName='source')
    #
    #     print('Images from test on target domain:')
    #     utils.displayImages(tgt_test_dataloader, imgName='target')

    # init models
    model_index = params.source_domain + '_' + params.target_domain
    feature_extractor = params.extractor_dict[model_index]
    class_classifier = params.class_dict[model_index]
    domain_classifier = params.domain_dict[model_index]


    if params.use_gpu:
        feature_extractor.cuda()
        class_classifier.cuda()
        domain_classifier.cuda()

    # 初始化损失函数,负对数似然损失（NLLLoss）
    class_criterion = nn.NLLLoss()
    domain_criterion = nn.NLLLoss()

    # 初始化优化器
    optimizer = optim.SGD([{'params': feature_extractor.parameters()},
                            {'params': class_classifier.parameters()},
                            {'params': domain_classifier.parameters()}], lr= params.lr, momentum= 0.9)


    total_loss_list = []
    class_loss_list = []
    domain_loss_list = []
    source_accu_list = []
    target_accu_list = []
    domain_accu_list = []
    # 在每个epoch中，调用train函数进行模型训练，调用test函数进行模型测试。根据设置，定期调用visualizePerformance函数进行可视化。
    for epoch in range(params.epochs):
        print('Epoch: {}'.format(epoch))

        # total_loss,class_loss,domain_loss =float(epoch)+2/2,float(epoch)+3/3,float(epoch)+5/5

        total_loss,class_loss,domain_loss = train.train(args.training_mode, feature_extractor, class_classifier, domain_classifier, class_criterion, domain_criterion,
                    src_train_dataloader, tgt_train_dataloader, optimizer, epoch)

        datastr2,source_accu,target_accu,domain_accu= test.test(feature_extractor, class_classifier, domain_classifier, src_test_dataloader,
                             tgt_test_dataloader)

        total_loss_list.append(total_loss)
        class_loss_list.append(class_loss)
        domain_loss_list.append(domain_loss)

        source_accu_list.append(source_accu)
        target_accu_list.append(target_accu)
        domain_accu_list.append(domain_accu)


        # 日志输出
        with open(params.log_path, 'a') as file:
            datastr1= f"{{\"total_loss\": {total_loss}, \"class_loss\": {class_loss}, \"domain_loss\": \"{domain_loss}\"}}"
            file.write(f"Epoch:{epoch}"+'\n')
            file.write(datastr1 + '\n')
            file.write(datastr2 + '\n')
            file.write("-----------------------------------------------------------" + '\n')

        # Plot embeddings periodically.
        if epoch % params.embed_plot_epoch == 0 and params.fig_mode is not None:
            visualizePerformance(feature_extractor, class_classifier, domain_classifier, src_test_dataloader,
                                 tgt_test_dataloader, imgName='embedding_' + str(epoch))



    utils.plt_loss(total_loss_list,"total_loss")
    utils.plt_loss(class_loss_list, "class_loss")
    utils.plt_loss(domain_loss_list,"domain_loss")

    utils.plt_loss(source_accu_list,"source_accu")
    utils.plt_loss(target_accu_list, "target_accu")
    utils.plt_loss(domain_accu_list,"domain_accu")


# 负责解析命令行参数
def parse_arguments(argv):
    """Command line parse."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--source_domain', type= str, default= 'MNIST', help= 'Choose source domain.')
    parser.add_argument('--target_domain', type= str, default= 'MNIST_M', help = 'Choose target domain.')
    parser.add_argument('--fig_mode', type=str, default=None, help='Plot experiment figures.')
    parser.add_argument('--save_dir', type=str, default=None, help='Path to save plotted images.')
    parser.add_argument('--training_mode', type=str, default='dann', help='Choose a mode to train the model.')
    parser.add_argument('--max_epoch', type=int, default=100, help='The max number of epochs.')
    parser.add_argument('--embed_plot_epoch', type= int, default=100, help= 'Epoch number of plotting embeddings.')
    parser.add_argument('--lr', type= float, default= 0.01, help= 'Learning rate.')

    return parser.parse_args()



# if __name__ == '__main__':
#     main(parse_arguments(sys.argv[1:]))
if __name__ == '__main__':
    args = argparse.Namespace(
        source_domain='MNIST',
        target_domain='MNIST_M',
        fig_mode = 'save' ,
        save_dir = None,
        training_mode='dann',
        max_epoch=110,
        embed_plot_epoch=10,
        lr=0.01
    )
    main(args)