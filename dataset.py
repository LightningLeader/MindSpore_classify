"""加载数据集"""
import os
import cv2
from ruamel import yaml
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore import dtype as mstype
import numpy as np



def getClasses(data_path, config_path):
    """
    加载路径下所有文件夹的名字然后进行升序排序。
    将第一个文件夹记作0，第二个文件夹记作1，依此类推。
    然后将文件夹名称和对应的标记写入config文件，用于制作数据集label。
    Args:
        data_path (str): 数据集路径
        config_path (str): 配置文件路径
    Returns:
        _type_: _description_
    """
    res_dict = {"classes":{}}
    data_list = os.listdir(data_path)
    data_list.sort()
    for data in data_list:
        res_dict["classes"][data] = data_list.index(data)
    with open(config_path, 'a+') as f:
        yaml.dump(res_dict, f, Dumper=yaml.RoundTripDumper)
        f.close()
    return res_dict["classes"]


def create_dataset(data_path, image_size, classDict=None, batch_size=24, repeat_num=1, training=True):
    """定义数据集"""
    
    # 默认是按照文件夹名称排序（字母顺序），每一个类被赋予一个从0开始的唯一索引
    data_set = ds.ImageFolderDataset(data_path, num_parallel_workers=8, shuffle=True, class_indexing=classDict)

    # 对数据进行增强操作
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    if training:
        trans = [
            # 裁剪、Decode、Resize
            CV.RandomCropDecodeResize(image_size, scale=(0.8, 1.0), ratio=(0.75, 1.333)),
            # 水平翻转
            CV.RandomHorizontalFlip(prob=0.5),
            # 随机旋转
            CV.RandomRotation(90),
            CV.Normalize(mean=mean, std=std),
            CV.HWC2CHW()
        ]
    else:
        trans = [
            CV.Decode(),
            CV.Resize((image_size, image_size)),
            CV.Normalize(mean=mean, std=std),
            CV.HWC2CHW()
        ]
    type_cast_op = C.TypeCast(mstype.int32)
    # 实现数据的map映射、批量处理和数据重复的操作
    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=8)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)
    data_set = data_set.batch(batch_size, drop_remainder=True)
    data_set = data_set.repeat(repeat_num)

    return data_set


class testDataset:
    """
    自定义的读取图片的类。
    只用于test使用，不返回label。
    返回：
    第一个值为用于分类的图
    第二个值为该图的ID，即名字和后缀
    第三个值为用于显示的图
    """
    def __init__(self, data_path):
        self.data_path = data_path
        self.imgList = os.listdir(self.data_path)

    def __getitem__(self, index):
        imgID = self.imgList[index]
        
        img = cv2.imread(os.path.join(self.data_path, imgID))
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        return img, imgID, img

    def __len__(self):
        return len(self.imgList)


if __name__ == "__main__":
    pass