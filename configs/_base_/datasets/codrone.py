# CODrone跨视角评估数据集配置
# 支持多种跨视角评估协议：AG（高度泛化）、AngG（角度泛化）、JG（联合泛化）

dataset_type = 'CODroneDataset'
data_root = 'data/CODrone/'
patch_data_root = 'data/split_ss_codrone_crossview/'
backend_args = None

view = 'AG-60'  # 可选视角：'STD'、'AG-30'、'AG-60'、'AG-100'、'AngG-30'、'AngG-90'、'JG-Extreme'

# 训练数据处理流水线 - 添加视角信息提取
train_pipeline = [
    # 加载图像
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    # 加载标注，使用四点框（qbox）格式
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    # 将四点框转换为旋转框（rbox）
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    # 图像缩放：图像缩放到1024x1024进行训练
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    # 随机翻转：仅水平翻转（垂直和对角线效果由RandomRotate覆盖）
    dict(
        type='mmdet.RandomFlip',
        prob=0.5,
        direction='horizontal'),
    # 打包检测输入，包含视角相关的meta信息
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# 验证/测试数据处理流水线
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(
        type='mmdet.Pad', size=(1024, 1024),
        pad_val=dict(img=(114, 114, 114))),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

test_pipeline = val_pipeline

# ============================================================
# 数据加载器
# ============================================================
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root=patch_data_root+view+'/',
        ann_file='train/annfiles/',
        data_prefix=dict(img_path='train/images/'),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=patch_data_root+view+'/',
        ann_file='val/annfiles/',
        data_prefix=dict(img_path='val/images/'),
        test_mode=True,
        pipeline=val_pipeline))

test_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=patch_data_root+view+'/',
        ann_file='test/annfiles/',
        data_prefix=dict(img_path='test/images/'),
        test_mode=True,
        pipeline=test_pipeline))

# ============================================================
# 标准评估配置（STD）- 使用原始CODrone划分
# ============================================================

# ============================================================
# 高度泛化评估配置（AG-X）
# AG-30: 训练时排除30m，测试30m
# AG-60: 训练时排除60m，测试60m  
# AG-100: 训练时排除100m，测试100m
# ============================================================

# ============================================================
# 角度泛化评估配置（AngG-X）
# AngG-30: 训练时排除30°，测试30°
# AngG-90: 训练时排除90°，测试90°
# ============================================================

# ============================================================
# 联合泛化评估配置（JG-Extreme）
# 训练时使用30m+90°的图像，测试100m+30°的图像
# 这是最极端的跨视角泛化测试
# ============================================================

# ============================================================
# 评估器配置
# ============================================================
val_evaluator = dict(type='DOTAMetric', metric='mAP', iou_thrs=[0.5, 0.75])
test_evaluator = dict(
    type='DOTAMetric', 
    format_only=False,
    metric='mAP', 
    iou_thrs=[0.5, 0.75],
    merge_patches=True,
    outfile_prefix='./work_dirs/codrone_crossview/Task1')
