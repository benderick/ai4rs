# 切片训练集
python tools/data/codrone/split/img_split.py --base-json \
  tools/data/codrone/split/split_configs/ss_codrone_train.json

# 切片验证集
python tools/data/codrone/split/img_split.py --base-json \
  tools/data/codrone/split/split_configs/ss_codrone_val.json

# 切片测试集
python tools/data/codrone/split/img_split.py --base-json \
  tools/data/codrone/split/split_configs/ss_codrone_test.json

# 跨视角
cd /home/zhangSHUO/futurama/openmmlab/ai4rs && python tools/data/codrone/split_crossview.py --protocol AG-30 --split train --use-symlink