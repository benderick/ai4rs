# Copyright (c) OpenMMLab. All rights reserved.
"""CODrone数据集类定义

CODrone是一个无人机视角下的目标检测数据集，包含12个类别：
- car: 汽车
- people: 行人
- motor: 摩托车/电动车
- truck: 卡车
- traffic-sign: 交通标志
- traffic-light: 交通灯
- boat: 船
- bus: 公交车
- bicycle: 自行车
- tricycle: 三轮车
- ship: 轮船
- bridge: 桥梁

标注格式与DOTA相同：x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
"""

from mmrotate.registry import DATASETS
from mmrotate.datasets.dota import DOTADataset


@DATASETS.register_module()
class CODroneDataset(DOTADataset):
    """CODrone数据集 - 无人机视角的旋转目标检测数据集
    
    该数据集包含4K分辨率的无人机图像，标注格式与DOTA相同。
    
    Note: 
        ``ann_file`` 是包含txt标注文件的文件夹路径
        图片后缀为 'jpg'
    
    Args:
        diff_thr (int): 难度阈值，大于此值的标注将被忽略。默认为100。
        img_suffix (str): 图片后缀。默认为 'jpg'。
    """

    METAINFO = {
        'classes': ('car', 'people', 'motor', 'truck', 'traffic-sign', 
                    'traffic-light', 'boat', 'bus', 'bicycle', 'tricycle',
                    'ship', 'bridge'),
        # palette 是用于可视化的颜色列表
        'palette': [
            (0, 255, 0),      # car - 绿色
            (255, 0, 0),      # people - 红色
            (255, 0, 255),    # motor - 紫色
            (255, 128, 0),    # truck - 橙色
            (255, 255, 0),    # traffic-sign - 黄色
            (0, 255, 128),    # traffic-light - 青绿色
            (0, 255, 255),    # boat - 青色
            (128, 0, 255),    # bus - 蓝紫色
            (255, 192, 203),  # bicycle - 粉色
            (165, 42, 42),    # tricycle - 棕色
            (0, 0, 255),      # ship - 蓝色
            (128, 128, 128),  # bridge - 灰色
        ]
    }

    def __init__(self,
                 diff_thr: int = 100,
                 img_suffix: str = 'jpg',
                 **kwargs) -> None:
        super().__init__(diff_thr=diff_thr, img_suffix=img_suffix, **kwargs)
