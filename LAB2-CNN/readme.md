# LAB2-CNN 使用说明

## 1. 基础训练

```bash
uv run python train.py --epochs 10 --batch-size 64 --learning-rate 0.001 --num-workers 4
```

如果只是想快速验证训练流程，可以适当减小训练轮数或采样比例：

```bash
uv run python train.py --epochs 3 --sample-ratio 0.2
```

训练完成后，输出将保存在 `outputs/时间戳/` 目录下，主要文件包括：

- `checkpoints/时间戳.pt`：验证集准确率最高时保存的模型参数
- `history.csv`：每个 epoch 的训练损失、训练准确率、验证损失和验证准确率
- `summary.json`：本次训练的配置与测试结果摘要
- `training_curves.png`：训练曲线图
- `confusion_matrix.png` / `confusion_matrix.csv`：混淆矩阵
- `misclassified_samples.png`：错分样本可视化
- `arguments.json`：该次训练的关键参数

## 2. 单独评估模型

如需对某个训练好的模型单独评估，可运行：

```bash
uv run python eval.py --checkpoint ./outputs/时间戳/checkpoints/模型文件.pt
```

如果模型训练时使用了非默认结构参数，还需要在评估时补充对应参数，例如：

```bash
uv run python eval.py \
  --checkpoint ./outputs/时间戳/checkpoints/模型文件.pt \
  --conv-blocks 4 \
  --kernel-size 3 \
  --pool-type max
```

评估结果默认保存在 `outputs/eval/` 下，包括测试集损失、准确率、混淆矩阵和错分样本图。

## 3. 运行对比实验

```bash
uv run python experiments.py --epochs 10 --batch-size 64 --learning-rate 0.001 --num-workers 4
```

如需快速筛选参数，也可以使用较小的采样比例：

```bash
uv run python experiments.py --epochs 5 --sample-ratio 0.5
```

脚本会自动完成三组对比实验：

- 网络深度：`conv_blocks = 2 / 3 / 4`
- 卷积核大小：`kernel_size = 3 / 5 / 7`
- 池化方式：`pool_type = max / avg`

实验结果会保存在 `outputs/experiment_时间戳/` 目录下，主要包括：

- `results/experiment_results.csv`：全部实验结果汇总
- `results/depth_experiment.csv`：网络深度实验结果
- `results/kernel_experiment.csv`：卷积核大小实验结果
- `results/pool_experiment.csv`：池化方式实验结果
- `figures/depth_curves.png`：深度对比曲线图
- `figures/kernel_curves.png`：卷积核大小对比曲线图
- `figures/pool_curves.png`：池化方式对比曲线图

每个实验子目录还会单独保存该组模型的训练曲线、混淆矩阵、错分样本和参数配置。
