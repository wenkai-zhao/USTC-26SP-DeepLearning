# 第一次实验：FNN应用

## 1. 环境准备

安装依赖：

```bash
pip install -r requirements.txt
```

## 2. 运行方式

### 2.1 单次训练

```bash
python train.py
```

可选参数示例：

```bash
python train.py --epochs 500 --learning-rate 0.001 --hidden-dims 64 32 --activation relu
```

### 2.2 三组对比实验（深度 / 学习率 / 激活函数）

```bash
python experiments.py
```

可选参数示例：

```bash
python experiments.py --epochs 500
```

## 3. 输出结果说明

- 每次运行会在 outputs 下新建时间戳目录，例如：experiment_YYYYMMDD_HHMMSS。
- 每个子实验会在该目录下创建独立 run 目录，保存：
  - checkpoints（最优模型参数）
  - history.csv（训练/验证损失曲线数据）
  - arguments.json（该次训练参数）
- 汇总结果在 results 目录：
  - depth_experiment.csv
  - learning_rate_experiment.csv
  - activation_experiment.csv
  - train_config.json
- 曲线图在 figures 目录（对应三组实验的 loss 曲线图）。
