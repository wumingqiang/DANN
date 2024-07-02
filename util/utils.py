import os

from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from train import params
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import time


# MNIST_M的Dataset
class MNIST_M_Dataset(Dataset):
    def __init__(self, data_folder, label_file, transform=None):
        self.data_folder = data_folder
        self.label_file = label_file
        self.transform = transform
        self.data = []
        self.labels = []
        self.load_data()
        print(len(self.data))

    def load_data(self):
        with open(self.label_file, 'r') as f:
            lines = f.readlines()

            for line in lines:
                filename, label = line.strip().split(' ')
                image_path = os.path.join(self.data_folder, filename)
                if os.path.exists(image_path):
                    self.data.append(image_path)
                    self.labels.append(int(label))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.data[index]
        label = self.labels[index]
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_train_loader(dataset):
    """
    Get test dataloader of source domain or target domain
    获得指定数据集的 dataloader
    :param:string
    :return: dataloader
    """
    if dataset == 'MNIST':
        # 定义数据转换,RGB图像->Tensor->归一化
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std, )
        ])

        # 创建MNIST数据集
        MNIST_dataset = datasets.MNIST(
            root=params.mnist_path,
            train=True,
            transform=transform,
            download=True
        )

        # 创建MNIST加载器
        dataloader = DataLoader(dataset= MNIST_dataset, batch_size= params.batch_size, shuffle= True)

    elif dataset == 'MNIST_M':
        # 随机裁剪->tensor->归一化
        transform = transforms.Compose([
            transforms.RandomCrop((28)),
            transforms.ToTensor(),
            transforms.Normalize(mean= params.dataset_mean, std= params.dataset_std)
        ])

        label_path = params.mnistm_train_label_path
        img_path = params.mnistm_train_image_path
        dataset = MNIST_M_Dataset(img_path,label_path,transform)

        # print(len(dataset))
        # print(dataset[0][0].shape)

        dataloader = DataLoader(dataset=dataset, batch_size=params.batch_size, shuffle=True)
    else:
        raise Exception('There is no (train)dataset named {}'.format(str(dataset)))

    return dataloader


def get_test_loader(dataset):
    """
    Get test dataloader of source domain or target domain
    获得指定数据集的 dataloader
    :param:string
    :return: dataloader
    """
    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=params.dataset_mean, std=params.dataset_std, )
        ])

        # 创建MNIST数据集
        dataset = datasets.MNIST(
            root=params.mnist_path,
            train=False,
            transform=transform,
            download=False
        )
        # 创建MNIST加载器
        dataloader = DataLoader(dataset=dataset, batch_size=params.batch_size, shuffle=True)

    elif dataset == 'MNIST_M':
        transform = transforms.Compose([
            transforms.RandomCrop((28)),
            transforms.ToTensor(),
            transforms.Normalize(mean= params.dataset_mean, std= params.dataset_std)
        ])

        label_path = params.mnistm_test_label_path
        img_path = params.mnistm_test_image_path
        mnistm_dataset = MNIST_M_Dataset(img_path,label_path,transform)

        dataloader = DataLoader(dataset=mnistm_dataset, batch_size=params.batch_size, shuffle=True)

    else:
        raise Exception('There is no (test)dataset named {}'.format(str(dataset)))

    return dataloader


# 调整学习率
def optimizer_scheduler(optimizer, p):
    """
        遍历参数组:优化器的 param_groups 属性包含了所有参数组。
        每个参数组是一个字典，包含了优化器的参数以及与这些参数相关的超参数
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.01 / (1. + 10 * p) ** 0.75

    return optimizer

# 绘制折线图
def plt_loss(data,title,x_label="x",y_label="y"):
    # 创建一个新的图形
    plt.figure()
    # 绘制折线图
    plt.plot(data)
    # 设置标题和标签
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # 保存图表为图片文件
    plt.savefig(f'{title}.png',)
    # 关闭当前图形，以免占用过多内存
    plt.close()

def displayImages(dataloader, size=8):
    """
    随机展示 size 张图片罢了
    :param dataloader:
    :param size:
    :return:
    """
    # 获取第一个批次的数据
    first_batch = next(iter(dataloader))
    # 获取第一个批次中的前几个样本
    num_samples = size
    first_batch_samples = first_batch[0][:num_samples]
    first_batch_labels = first_batch[1][:num_samples]

    images = []
    for i in range(size):
        image = first_batch_samples[i]
        label = first_batch_labels[i]
        image = image.squeeze().numpy()  # 将图像从张量转换为NumPy数组
        image = np.transpose(image, (1, 2, 0))  # 重新排列维度
        image = (image * 255).astype(np.uint8)  # 将像素值恢复到0-255范围内
        image = Image.fromarray(image)  # 创建PIL图像对象

        # 在图像上绘制标签
        draw = ImageDraw.Draw(image)
        label_text = str(label.item())
        label_font = ImageFont.truetype("arial.ttf", 12)  # 指定字体和字号
        label_color = (255, 0, 0)  # 标签的颜色（红色）
        label_position = (5, 5)  # 标签的位置
        draw.text(label_position, label_text, fill=label_color, font=label_font)

        images.append(image)

    # 将图像分为多行，每行最多显示8个图像
    rows = [images[i:i+8] for i in range(0, size, 8)]

    # 将每一行的图像进行垂直堆叠
    stacked_rows = [np.hstack(row) for row in rows]

    # 将所有行的图像进行水平堆叠
    stacked_image = np.vstack(stacked_rows)

    # 显示堆叠后的图像
    stacked_image = Image.fromarray(stacked_image)
    stacked_image.show()


def plot_embedding(X, y, d, title=None, imgName=None):
    """
    Plot an embedding X with the class label y colored by the domain d.

    :param X: embedding
    :param y: label
    :param d: domain
    :param title: title on the figure
    :param imgName: the name of saving image
    :return:
    """
    if params.fig_mode is None:
        return
    # normalization
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)

    for i in range(X.shape[0]):
        # plot colored number
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.bwr(d[i]/1.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])

    # If title is not given, we assign training_mode to the title.
    if title is not None:
        plt.title(title)
    else:
        plt.title(params.training_mode)

    if params.fig_mode == 'display':
        print("display")
        # Directly display if no folder provided.
        plt.show()

    if params.fig_mode == 'save':
        print("save")
        # Check if folder exist, otherwise need to create it.
        folder = os.path.abspath(params.save_dir)

        if not os.path.exists(folder):
            os.makedirs(folder)

        print("imgName:",imgName)
        if imgName is None:
            imgName = 'plot_embedding' + str(int(time.time()))

        # Check extension in case.
        if not (imgName.endswith('.jpg') or imgName.endswith('.png') or imgName.endswith('.jpeg')):
            imgName = os.path.join(folder, imgName + '.jpg')

        print('Saving ' + imgName + ' ...')
        plt.savefig(imgName)
        plt.close()

#
# ### test
# mnist_trainloader = get_train_loader("MNIST")
# # mnist_testloader = get_test_loader("MNIST")
#
# mnist_m_trainloader = get_train_loader("MNIST_M")
# # mnist_m_testloader = get_test_loader("MNIST_M")
# #
# displayImages(mnist_trainloader, size = 32)
# displayImages(mnist_m_trainloader, size = 32)