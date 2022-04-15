import os
import argparse
import yaml
import pandas as pd

from dataset import create_dataset, getClasses
from utils import EvalCallBack, apply_eval, filter_checkpoint_parameter_by_list

import mindspore.nn as nn
from mindspore.train.callback import TimeMonitor
from mindspore import Model, context, load_checkpoint, load_param_into_net
from resnet.src.resnet import resnet50


def parse_args():
    parser = argparse.ArgumentParser()
    
    # 工程文件名字
    parser.add_argument("--name", default="resnet50_classify", help="The name of project.")
    # 数据集名称
    parser.add_argument('--dataset', default='Canidae', help='dataset name')
    parser.add_argument("--epochs", default=200, type=int, metavar='N')
    parser.add_argument('--batch_size', default=16, type=int, metavar='N')

    # 预训练权重，参数填预训练文件的路径
    parser.add_argument('--pre_ckpt', default="./pre_ckpt/resnet50.ckpt")
    # 是否删除预训练模型的全连接层
    parser.add_argument("--delfc_flag", default=True)
    
    # 输入图片的channels，默认是RGB三通道图片
    parser.add_argument('--input_channels', default=3, type=int, help='input channels')
    # 类别个数
    parser.add_argument('--num_classes', default=20, type=int, help='number of classes')
    # 输入图像的尺寸
    parser.add_argument('--image_size', default=128, type=int, help='image size')

    # 优化器
    parser.add_argument('--optimizer', default='Adam')
    # 损失函数
    parser.add_argument('--loss', default='SoftmaxCrossEntropyWithLogits')
    
    parser.add_argument('--dataset_sink_mode', default=False)


    config = parser.parse_args()
    return config


def main():
    config = vars(parse_args())
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    # 创建模型工程文件夹，用来保存训练文件
    if config["name"] is None:
        config['name'] = "test"
    os.makedirs(os.path.join("models", config["name"]), exist_ok=True)

    # 创建config文件
    config_path = 'models/%s/config.yml' % config['name']
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    # 创建log
    columns = ["epoch", "train_loss", "val_acc", "best_acc"]
    dt = pd.DataFrame(columns=columns)
    dt.to_csv("models/%s/log.csv"%config["name"], index=0)

    # 创建网络
    model = resnet50(config["num_classes"])

    # 是否删除预训练模型全连接层的参数
    delfc_flag = config["delfc_flag"]
    # 是否加载预训练权重
    if config["pre_ckpt"] is not None:
        # 加载预训练模型
        param_dict = load_checkpoint(config["pre_ckpt"])
        if delfc_flag:
            # 获取全连接层的名字
            filter_list = [x.name for x in model.end_point.get_parameters()]
            # 删除预训练模型的全连接层
            filter_checkpoint_parameter_by_list(param_dict, filter_list)
        # 给网络加载参数
        load_param_into_net(model, param_dict)

    # 定义损失函数
    if config['loss'] == "SoftmaxCrossEntropyWithLogits":
        criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # 定义优化器
    if config['optimizer'] == "Adam":
        optimizer = nn.Adam(params=model.trainable_params(), learning_rate=0.001)
    elif config['optimizer'] == "Momentum":
        optimizer = nn.Momentum(params=model.trainable_params(), learning_rate=0.001, momentum=0.9)

    # 实例化模型
    model = Model(model, criterion, optimizer, metrics={"Accuracy":nn.Accuracy()})

    # 加载数据集
    train_data_path = "inputs/%s/train" % config["dataset"]
    val_data_path = "inputs/%s/val" % config["dataset"]
    # 获取label，就是获取所有种类名称将其与数字对应
    classesDict = getClasses(train_data_path, config_path)
    train_ds = create_dataset(train_data_path,
                              image_size=config["image_size"],
                              batch_size=config["batch_size"],
                              classDict=classesDict,
                              training=True)
    val_ds = create_dataset(val_data_path,
                            image_size=config["image_size"],
                            batch_size=config["batch_size"],
                            classDict=classesDict,
                            training=False)

    # 验证集回馈
    eval_param_dict = {"model":model,
                       "dataset":val_ds,
                       "metrics_name":"Accuracy",
                       "dataset_sink_mode":config["dataset_sink_mode"]}
    eval_cb = EvalCallBack(apply_eval,
                           eval_param_dict,
                           config=config,
                           ckpt_directory="./models/%s/" % config['name'])

    # 模型训练
    model.train(config["epochs"],
                train_ds,
                callbacks=[eval_cb, TimeMonitor()],
                dataset_sink_mode=config["dataset_sink_mode"])


if __name__ == "__main__":
    main()
