from models import models

# utility params
fig_mode = 'save'

embed_plot_epoch = 10
# embed_imgName = 'emadding.png'

# model params
use_gpu = True
dataset_mean = (0.5, 0.5, 0.5)
dataset_std = (0.5, 0.5, 0.5)

batch_size = 512
epochs = 1000
gamma = 10
# 标签损失和域损失的比重
theta = 1


# 数据根目录和一些常见的目录
data_root = r'E:\Python_projects\DANN\Data_xia'


mnist_path = data_root + '/MNIST'

mnistm_path = data_root + '/MNIST_M'
mnistm_train_label_path = mnistm_path + "/mnist_m_train_labels.txt"
mnistm_train_image_path = mnistm_path + "/mnist_m_train"

mnistm_test_label_path = mnistm_path + "/mnist_m_test_labels.txt"
mnistm_test_image_path = mnistm_path + "/mnist_m_test"
save_dir = './experiment'

log_path = 'log.json'


# specific dataset params
# 定义了三个字典变量 extractor_dict、class_dict 和 domain_dict，用于存储不同数据集的模型

extractor_dict = {'MNIST_MNIST_M': models.Extractor(),
                  }

class_dict = {'MNIST_MNIST_M': models.Class_classifier(),
              }

domain_dict = {'MNIST_MNIST_M': models.Domain_classifier(),
               }
