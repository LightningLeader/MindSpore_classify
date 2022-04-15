import numpy as np
import argparse
from ruamel import yaml
import cv2
import shutil
import os


from matplotlib import pyplot as plt
from mindspore.ops import ExpandDims
import mindspore.nn as nn
from mindspore import Tensor, Model, load_checkpoint, load_param_into_net
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV


from dataset import testDataset
from resnet.src.resnet import resnet50


def parse_args():
    parser = argparse.ArgumentParser()

    # 填写要验证模型名字
    parser.add_argument('--name', default="resnet50_classify")
    
    # 填写测试文件夹路径
    parser.add_argument("--test_path", default="testImg")
    
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # 读取配置文件
    with open('models/%s/config.yml' % args.name, 'r', encoding='utf-8') as f:
        # 字典类型
        config = yaml.load(f.read(), Loader=yaml.Loader)
    print("Load %s config successfully!" % args.name)


    model = resnet50(config["num_classes"])

    # 加载模型文件
    param_dict = load_checkpoint("./models/%s/best.ckpt"%config["name"])
    load_param_into_net(model, param_dict)


    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    # 实例化模型
    model = Model(model, criterion, metrics={"Accuracy":nn.Accuracy()})

    # 测试集数据
    # 测试集路径
    test_data_path = args.test_path
    
    # classes格式
    classesDict = config["classes"]
    # 将字典的键值调换顺序
    class_name = dict(zip(classesDict.values(), classesDict.keys()))

    # 读取测试数据并做预处理
    testData = testDataset(test_data_path)
    # image用于分类处理；imgID为图片名字和后缀；imageInit用于显示原图
    dataset = ds.GeneratorDataset(testData, ["image", "imgID", "imageInit"])
    
    # 图像预处理，主要是归一化和通道交换顺序
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    dataset = dataset.map(operations=[CV.Normalize(mean=mean, std=std), CV.HWC2CHW()],
                          input_columns="image",
                          num_parallel_workers=8)
    # dataset = dataset.batch(config["batch_size"], drop_remainder=True)
    
    # 用于存放预测结果
    pred_list = []
    # 存放显示图片
    img_list = []
    # 图片名字
    imgID_list = []
    # 扩充维度
    expand = ExpandDims()
    for data in dataset.create_dict_iterator():
        # 将图像添加到待显示图片列表中
        img_list.append(data["imageInit"].asnumpy())
        imgID_list.append(str(data["imgID"]))
        # 使用处理后的图像来预测
        image = data["image"].asnumpy()
        # 扩充维度
        img = expand(Tensor(image), 0)
        output = model.predict(img)
        pred = np.argmax(output.asnumpy(), axis=1)[0]
        pred_list.append(pred)
    # # 可视化模型预测
    plt.figure(figsize=(12, 5))
    # 显示24张图
    for i in range(len(img_list)):
        plt.subplot(3, 10, i+1)
        plt.title('{}'.format(class_name[pred_list[i]]))
        picture_show = img_list[i]/np.amax(img_list[i])
        picture_show = np.clip(picture_show, 0, 1)
        plt.imshow(picture_show)
        plt.axis('off')
    plt.show()

    if os.path.exists(os.path.join("outputs", config["name"])):
        shutil.rmtree(os.path.join("outputs", config["name"]))
        print("目录存在，已删除")
    # 根据类别创建对应文件夹
    for key in classesDict.keys():
        os.makedirs(os.path.join("outputs", config["name"], key), exist_ok=True)
    print("成功建立分类文件夹")

    # 根据分类结果将图片复制到对应文件夹
    for index, imgID in enumerate(imgID_list):
        shutil.copy(os.path.join(test_data_path, imgID),
                    os.path.join("outputs", config["name"], class_name[pred_list[index]], imgID))
    print("分类完成")


if __name__ == "__main__":
    main()
