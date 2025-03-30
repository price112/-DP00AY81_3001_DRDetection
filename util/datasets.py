# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.datasets.folder import default_loader
from typing import Dict, List, Tuple

import os

class MergedImageFolder(datasets.ImageFolder):
    def __init__(self, root: str, merge_map: Dict[str, List[str]] = None, **kwargs):
        """
        Args:
            root: 数据集根目录
            merge_map: 合并规则字典，格式如{'class_name': ['folder1', 'folder2']}
            **kwargs: 其他ImageFolder参数（transform等）
        """
        self.merge_map = merge_map if merge_map else {}
        super().__init__(root, **kwargs)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        # 获取所有有效子文件夹
        folders = [d.name for d in os.scandir(directory) if d.is_dir()]

        # 构建文件夹到目标类的映射
        folder_to_class = {}
        for cls_name, folder_list in self.merge_map.items():
            for folder in folder_list:
                if folder in folders:
                    folder_to_class[folder] = cls_name

        # 处理未合并的文件夹
        for folder in folders:
            if folder not in folder_to_class:
                folder_to_class[folder] = folder  # 保持独立类别

        # 生成最终类别列表
        class_names = sorted(list(set(folder_to_class.values())))
        class_to_idx = {cls: i for i, cls in enumerate(class_names)}

        # 创建原始文件夹到索引的映射
        merged_class_to_idx = {
            folder: class_to_idx[target_cls]
            for folder, target_cls in folder_to_class.items()
        }

        return class_names, merged_class_to_idx



def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, is_train)

    if args.cls_task == 5:
        merge_rules_5 = {
            0 : ["anodr"],
            1 : ["bmilddr"],
            2 : ["cmoderatedr"],
            3 : ["dseveredr"],
            4 : ["eproliferativedr"]
        }

        dataset = MergedImageFolder(
            root=root,
            merge_map = merge_rules_5,
            transform=transform
        )

        args.nb_classes = 5
        assert len(dataset.classes) == 5


    elif args.cls_task == 3:

        merge_rules_3 = {
            0 : ["anodr"],
            1 : ["bmilddr", "cmoderatedr","dseveredr"],
            2 : ["eproliferativedr"]
        }

        dataset = MergedImageFolder(
            root=root,
            merge_map = merge_rules_3,
            transform=transform
        )

        args.nb_classes = 3
        assert len(dataset.classes) == 3

    elif args.cls_task == 2:

        merge_rules_2 = {
            0 : ["anodr"],
            1 : ["bmilddr", "cmoderatedr","dseveredr","eproliferativedr"]
        }

        dataset = MergedImageFolder(
            root=root,
            merge_map = merge_rules_2,
            transform=transform
        )

        args.nb_classes = 2
        assert len(dataset.classes) == 2
    else:
        raise ValueError('Unknown cls_task')

    return dataset

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train == 'train':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

