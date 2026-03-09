# Astro 项目阶段记录（v2 backbone + frozen redshift probe）

更新时间：2026-03-09

## 1. 当前主线结论

当前项目的主线已经明确为两阶段：

1. **统一天文表征预训练**：使用 `v2` 多模态自监督 backbone 训练统一表征。
2. **下游任务解耦验证**：冻结 backbone，只训练小任务头，验证表征是否真的有用。

当前已经完成的是：
- `v2` backbone 训练完成，并选定 `final checkpoint` 作为后续统一使用的表征底座。
- 基于 `v2 final checkpoint` 实现并跑通了一个完整的 **frozen redshift probe** 流程。
- 当前下游验证任务是：**redshift regression**。
- 当前 probe 默认特征是：**`z_shared`**。
- 当前 probe 默认严格 **禁用 metadata**，避免 `log1p(z)` 标签泄漏。

当前默认使用的 backbone checkpoint：
- `astro_sdss_v2_20gb_curated/experiments/full_20260306_160018_tty/checkpoint-final.pth`

---

## 2. 当前版本训练方案是什么

### 2.1 Backbone 训练方案（已完成）

当前 backbone 仍然是你原来的 `v2` 路线，不是 `v3`。

配置入口：
- `configs/train_astro_sdss_v2.yaml`
- `configs/dataset/astro_sdss_pair_v2.yaml`
- `configs/model/astro_mapanything_v2.yaml`
- `configs/loss/astro_bidirectional_v2.yaml`

核心结构：
- 输入：`5-band 图像 + 1D 光谱 (+ metadata，仅预训练阶段)`
- 图像分支：基于 `DINOv2` patch7 改造的图像 encoder
- 光谱分支：1D spectrum encoder
- 融合层：fusion transformer
- 输出表征：
  - `z_img`
  - `z_spec`
  - `z_shared`
  - `z_nuis`

三阶段训练策略：
- **Stage A（20 epoch）**：只训练 `both`
- **Stage B（80 epoch）**：混合训练 `image-only / spectrum-only / both`
- **Stage C（20 epoch）**：进一步增加缺模态训练占比，强化 shared representation

输入模式采样概率：
- Stage B：`(0.25, 0.15, 0.60)` 对应 `image-only / spectrum-only / both`
- Stage C：`(0.30, 0.15, 0.55)` 对应 `image-only / spectrum-only / both`

损失是组合式多任务自监督，主要包括：
- 跨模态预测损失
- 光谱自重建损失
- 图像自重建损失
- 跨模态 shared 对齐损失
- 一致性损失
- nuisance / disentangle 相关损失
- auxiliary spectrum loss
- latent self-distillation

### 2.2 当前下游方案（已完成）

当前已经实现的是 **冻结 backbone + 小 MLP 任务头** 的 probe 方案：

- backbone：固定使用 `v2 final checkpoint`
- 特征：默认使用 `z_shared`
- 输入模式：
  - `image-only`
  - `spectrum-only`
  - `both`
- 目标：预测 `log1p(z)`，评估时还原到 `z`
- metadata：默认强制禁用

probe 头结构：
- `768 -> 256 -> 64 -> 1`
- 激活：`GELU`
- Dropout：`0.1`

训练超参：
- optimizer：`AdamW`
- lr：`1e-3`
- weight decay：`1e-4`
- batch size：`64`
- max epochs：`100`
- early stopping patience：`10`

---

## 3. 当前已经完成了什么

### 3.1 已完成的主要工作

1. 选定并固定后续主用 checkpoint：
   - `v2 final checkpoint`
   - 不再使用 `best` 作为下游验证主入口

2. 明确项目后续主线：
   - 先训练统一天文表征
   - 再解耦做各类下游任务

3. 实现了 frozen redshift probe 工具链：
   - 导出 frozen backbone 特征
   - 训练小型 redshift probe head
   - 自动保存指标、预测结果、误差图和汇总表

4. 已经跑通两类实验：
   - smoke test（8/8 train/val 小样本）
   - full probe（743/83 train/val 全量）

5. 新增了结果分析 notebook：
   - 可直接读取 `summary.csv`、`metrics.json`、`predictions.csv`
   - 画对比柱状图、散点图、误差图

### 3.2 当前完成后的判断

从当前结果看：
- `z_shared` 的 frozen representation **确实包含 redshift 信息**，说明 backbone 不是无效表征。
- 当前 `spectrum-only` 表现最好，说明 `photo-z` 这件事上 backbone 仍然更依赖光谱信息。
- 当前 `both` 没有超过 `spectrum-only`，说明 **shared representation 的多模态增益还不明显**。
- 当前这套结果能支持“统一表征可用”的结论，但也明确暴露出图像分支对 redshift probe 的增益还不够强。

---

## 4. 本轮新增的代码改动

### 4.1 新增文件

1. Probe 工具模块：
- `mapanything/utils/astro_probe.py`

2. 特征导出脚本：
- `scripts/export_astro_probe_features.py`

3. Frozen probe 训练脚本：
- `scripts/train_astro_redshift_probe.py`

4. 一键运行总控脚本：
- `scripts/run_astro_redshift_probe.py`

5. 结果分析 notebook：
- `notebooks/astro_redshift_probe_results.ipynb`

6. 基础测试文件：
- `tests/test_astro_probe_metrics.py`

### 4.2 每个新增文件的用途

#### `mapanything/utils/astro_probe.py`
统一封装：
- experiment config / checkpoint 加载
- dataset 构建
- frozen feature 结果读写
- redshift 指标计算
- 汇总图绘制

#### `scripts/export_astro_probe_features.py`
作用：
- 加载 `v2 final checkpoint`
- 遍历 train / val
- 导出三种模式下的 frozen features：
  - `image-only`
  - `spectrum-only`
  - `both`
- 导出字段包括：
  - `sample_id`
  - `z`
  - `ra`
  - `dec`
  - `sn_median_r`
  - `target_log1p_z`
  - `z_shared`
  - `z_img`
  - `z_spec`
  - `latent_shared_proj` 等

#### `scripts/train_astro_redshift_probe.py`
作用：
- 从导出的 frozen feature 文件中读取指定模式数据
- 只训练小 MLP 任务头
- 保存：
  - `best_probe_head.pt`
  - `metrics.json`
  - `predictions.csv`
  - `history.csv`
  - `train_curve.png`
  - `scatter_z_true_vs_pred.png`
  - `residual_hist.png`
  - `residual_vs_z.png`
  - `residual_vs_sn_median_r.png`

#### `scripts/run_astro_redshift_probe.py`
作用：
- 一条命令串起来：
  1. 导出特征
  2. 分别训练 `image-only / spectrum-only / both`
  3. 输出总汇总表 `summary.csv`

#### `notebooks/astro_redshift_probe_results.ipynb`
作用：
- 读取 probe 输出目录
- 展示 summary 表格
- 画每个 mode 的核心结果图
- 方便后续论文图/组会图整理

---

## 5. 新增的训练 / 运行命令

以下命令默认在仓库根目录 `/root/wyj/map-anything-sdss` 下执行。

### 5.1 导出 frozen 特征

```bash
conda run -n mapanything-offline-310 python scripts/export_astro_probe_features.py \
  --experiment-dir astro_sdss_v2_20gb_curated/experiments/full_20260306_160018_tty \
  --checkpoint checkpoint-final.pth \
  --output-dir astro_sdss_v2_20gb_curated/probes/redshift_probe/features \
  --modes image-only,spectrum-only,both \
  --batch-size 16 \
  --num-workers 4 \
  --device cuda \
  --disable-metadata \
  --dino-local-repo /root/wyj/dinov2 \
  --dino-local-ckpt /root/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth
```

### 5.2 单独训练某一个 probe mode

例如训练 `both`：

```bash
conda run -n mapanything-offline-310 python scripts/train_astro_redshift_probe.py \
  --features-root astro_sdss_v2_20gb_curated/probes/redshift_probe/features \
  --mode both \
  --output-dir astro_sdss_v2_20gb_curated/probes/redshift_probe/runs/both \
  --input-key z_shared \
  --batch-size 64 \
  --epochs 100 \
  --patience 10 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --dropout 0.1 \
  --hidden-dims 256,64 \
  --seed 42 \
  --device cuda
```

### 5.3 一键完整运行整个 frozen redshift probe

```bash
conda run -n mapanything-offline-310 python scripts/run_astro_redshift_probe.py \
  --experiment-dir astro_sdss_v2_20gb_curated/experiments/full_20260306_160018_tty \
  --checkpoint checkpoint-final.pth \
  --probe-root astro_sdss_v2_20gb_curated/probes/redshift_probe \
  --modes image-only,spectrum-only,both \
  --feature-key z_shared \
  --batch-size-export 16 \
  --batch-size-train 64 \
  --epochs 100 \
  --patience 10 \
  --num-workers 4 \
  --device cuda \
  --dino-local-repo /root/wyj/dinov2 \
  --dino-local-ckpt /root/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth
```

### 5.4 打开结果 notebook

```bash
jupyter notebook notebooks/astro_redshift_probe_results.ipynb
```

---

## 6. 当前实验结果

### 6.1 Full probe 结果（全量）

结果汇总文件：
- `astro_sdss_v2_20gb_curated/probes/redshift_probe/summary.csv`

全量数据规模：
- train：`743`
- val：`83`

结果如下：

| mode | input_key | best_epoch | MAE(z) | RMSE(z) | median |Δz|/(1+z) | outlier rate | Pearson | Spearman |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| image-only | z_shared | 30 | 0.027434 | 0.037691 | 0.018217 | 0.000000 | 0.884902 | 0.705911 |
| spectrum-only | z_shared | 36 | 0.025436 | 0.035303 | 0.015998 | 0.000000 | 0.908785 | 0.681688 |
| both | z_shared | 26 | 0.029204 | 0.038288 | 0.019223 | 0.000000 | 0.885878 | 0.695290 |

当前结论：
- `spectrum-only` 最好
- `image-only` 次之
- `both` 没有超过 `spectrum-only`

这说明：
- `z_shared` 里已经有可用 redshift 信息
- 但当前 unified representation 对 `photo-z` 的多模态增益还不明显

### 6.2 Smoke test 结果（小样本流程验证）

结果汇总文件：
- `astro_sdss_v2_20gb_curated/probes/redshift_probe_smoke/summary.csv`

小样本规模：
- train：`8`
- val：`8`

结果如下：

| mode | input_key | best_epoch | MAE(z) | RMSE(z) | median |Δz|/(1+z) | outlier rate |
|---|---|---:|---:|---:|---:|---:|
| image-only | z_shared | 3 | 0.083681 | 0.101216 | 0.069279 | 0.125 |
| spectrum-only | z_shared | 3 | 0.088669 | 0.104914 | 0.073331 | 0.125 |
| both | z_shared | 3 | 0.085500 | 0.104555 | 0.072391 | 0.125 |

作用：
- 只用于验证脚本链路、导出逻辑、训练逻辑、绘图逻辑是通的
- 不用于科学结论

---

## 7. 当前结果文件都在哪里

### 7.1 Full probe 根目录
- `astro_sdss_v2_20gb_curated/probes/redshift_probe/`

### 7.2 Frozen feature 文件
- `astro_sdss_v2_20gb_curated/probes/redshift_probe/features/train_image_only.npz`
- `astro_sdss_v2_20gb_curated/probes/redshift_probe/features/train_spectrum_only.npz`
- `astro_sdss_v2_20gb_curated/probes/redshift_probe/features/train_both.npz`
- `astro_sdss_v2_20gb_curated/probes/redshift_probe/features/val_image_only.npz`
- `astro_sdss_v2_20gb_curated/probes/redshift_probe/features/val_spectrum_only.npz`
- `astro_sdss_v2_20gb_curated/probes/redshift_probe/features/val_both.npz`
- `astro_sdss_v2_20gb_curated/probes/redshift_probe/features/export_metadata.json`

### 7.3 各 mode 训练结果目录
- `astro_sdss_v2_20gb_curated/probes/redshift_probe/runs/image_only/`
- `astro_sdss_v2_20gb_curated/probes/redshift_probe/runs/spectrum_only/`
- `astro_sdss_v2_20gb_curated/probes/redshift_probe/runs/both/`

每个目录下都有：
- `best_probe_head.pt`
- `metrics.json`
- `predictions.csv`
- `history.csv`
- `history.json`
- `train_curve.png`
- `scatter_z_true_vs_pred.png`
- `residual_hist.png`
- `residual_vs_z.png`
- `residual_vs_sn_median_r.png`
- `z_bins.csv`
- `sn_bins.csv`

### 7.4 汇总结果
- `astro_sdss_v2_20gb_curated/probes/redshift_probe/summary.csv`
- `astro_sdss_v2_20gb_curated/probes/redshift_probe/summary.json`
- `astro_sdss_v2_20gb_curated/probes/redshift_probe/summary_metrics.png`

---

## 8. 当前已知注意事项

1. **probe 阶段默认必须禁用 metadata**
   - 因为当前 metadata 中含有 `log1p(z)`
   - 如果开 metadata，会直接造成 redshift 标签泄漏

2. 当前 frozen probe 的目标不是追求最强指标，而是验证：
   - `v2 final` 的统一表征是否真的有信息
   - 各模态输入对 shared representation 的贡献如何

3. 当前结果已经证明：
   - frozen `z_shared` 不是空表征
   - 但图像分支对 redshift 的额外帮助还不够强

4. 当前 `both < spectrum-only` 不表示系统坏了，更多说明：
   - 在 `photo-z` 这个下游上，当前 shared representation 还偏向光谱主导

---

## 9. 下面该干嘛

建议的下一步优先级如下：

### 第一优先级：做表征剖析对比

在当前 frozen probe 框架下继续补以下实验：

1. `z_shared` vs `z_img` vs `z_spec`
   - 比较三类表征对 redshift 的可读性
   - 判断当前 shared 是否损失了光谱专有信息

2. `linear probe` vs `MLP probe`
   - 判断信息是线性可分还是需要非线性头才能读出来

3. `shared only` vs `concat(shared, private)`
   - 比如：`[z_shared, z_spec]`
   - 看是否能证明 shared + private 更合适

### 第二优先级：补第二个下游任务

推荐从实现最容易的开始：

1. **分类任务**（如果当前数据里有类别标签可以整理出来）
2. **跨模态检索 / matching**
3. **基于表征的相似样本检索**

### 第三优先级：为论文准备展示材料

如果目标是说服编委，后面需要补的不只是一个 probe 分数，而是一套“表征能力证据”：

1. frozen probe 结果表
2. 不同表征层对比表
3. 检索可视化
4. t-SNE / UMAP 投影
5. 随 `z` 与 `sn_median_r` 的误差分布图

---

## 10. 推荐下一步命令

如果下一步要继续做表征剖析，建议从下面两个方向开始：

### 10.1 先测 `z_spec`

```bash
conda run -n mapanything-offline-310 python scripts/train_astro_redshift_probe.py \
  --features-root astro_sdss_v2_20gb_curated/probes/redshift_probe/features \
  --mode spectrum-only \
  --output-dir astro_sdss_v2_20gb_curated/probes/redshift_probe_zspec/runs/spectrum_only \
  --input-key z_spec \
  --batch-size 64 \
  --epochs 100 \
  --patience 10 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --dropout 0.1 \
  --hidden-dims 256,64 \
  --seed 42 \
  --device cuda
```

### 10.2 再测 `z_img`

```bash
conda run -n mapanything-offline-310 python scripts/train_astro_redshift_probe.py \
  --features-root astro_sdss_v2_20gb_curated/probes/redshift_probe/features \
  --mode image-only \
  --output-dir astro_sdss_v2_20gb_curated/probes/redshift_probe_zimg/runs/image_only \
  --input-key z_img \
  --batch-size 64 \
  --epochs 100 \
  --patience 10 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --dropout 0.1 \
  --hidden-dims 256,64 \
  --seed 42 \
  --device cuda
```

---

## 11. 后续维护约定

从现在开始，后续如再有：
- 新增实验
- 新增脚本
- 新增结果目录
- 重要结论变化
- 训练命令更新

建议都继续追加到这个文件里，作为统一项目日志。


---

## 12. 2026-03-09 新增实验：`z_shared / z_img / z_spec` 表征对比

### 12.1 这轮实验的目标

这一轮实验的目标是回答：
- 当前 redshift probe 里，为什么 `z_shared` 没有超过 `spectrum-only`？
- 问题到底出在：
  - `shared representation` 本身压缩过强，还是
  - 图像分支没有提供有效信息？

因此，本轮新增了对比实验：
- `image-only + z_img`
- `spectrum-only + z_spec`
- `both + z_img`
- `both + z_spec`

并和已存在 baseline 做对比：
- `image-only + z_shared`
- `spectrum-only + z_shared`
- `both + z_shared`

### 12.2 本轮新增运行命令

#### `image-only + z_img`

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n mapanything-offline-310 python scripts/train_astro_redshift_probe.py \
  --features-root astro_sdss_v2_20gb_curated/probes/redshift_probe/features \
  --mode image-only \
  --output-dir astro_sdss_v2_20gb_curated/probes/redshift_probe_feature_ablation/runs/image_only_z_img \
  --input-key z_img \
  --batch-size 64 \
  --epochs 100 \
  --patience 10 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --dropout 0.1 \
  --hidden-dims 256,64 \
  --seed 42 \
  --device cuda
```

#### `spectrum-only + z_spec`

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n mapanything-offline-310 python scripts/train_astro_redshift_probe.py \
  --features-root astro_sdss_v2_20gb_curated/probes/redshift_probe/features \
  --mode spectrum-only \
  --output-dir astro_sdss_v2_20gb_curated/probes/redshift_probe_feature_ablation/runs/spectrum_only_z_spec \
  --input-key z_spec \
  --batch-size 64 \
  --epochs 100 \
  --patience 10 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --dropout 0.1 \
  --hidden-dims 256,64 \
  --seed 42 \
  --device cuda
```

#### `both + z_img`

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n mapanything-offline-310 python scripts/train_astro_redshift_probe.py \
  --features-root astro_sdss_v2_20gb_curated/probes/redshift_probe/features \
  --mode both \
  --output-dir astro_sdss_v2_20gb_curated/probes/redshift_probe_feature_ablation/runs/both_z_img \
  --input-key z_img \
  --batch-size 64 \
  --epochs 100 \
  --patience 10 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --dropout 0.1 \
  --hidden-dims 256,64 \
  --seed 42 \
  --device cuda
```

#### `both + z_spec`

```bash
CUDA_VISIBLE_DEVICES=3 conda run -n mapanything-offline-310 python scripts/train_astro_redshift_probe.py \
  --features-root astro_sdss_v2_20gb_curated/probes/redshift_probe/features \
  --mode both \
  --output-dir astro_sdss_v2_20gb_curated/probes/redshift_probe_feature_ablation/runs/both_z_spec \
  --input-key z_spec \
  --batch-size 64 \
  --epochs 100 \
  --patience 10 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --dropout 0.1 \
  --hidden-dims 256,64 \
  --seed 42 \
  --device cuda
```

### 12.3 本轮结果文件位置

本轮新增目录：
- `astro_sdss_v2_20gb_curated/probes/redshift_probe_feature_ablation/`

汇总文件：
- `astro_sdss_v2_20gb_curated/probes/redshift_probe_feature_ablation/summary.csv`
- `astro_sdss_v2_20gb_curated/probes/redshift_probe_feature_ablation/combined_summary.csv`

具体运行结果目录：
- `astro_sdss_v2_20gb_curated/probes/redshift_probe_feature_ablation/runs/image_only_z_img/`
- `astro_sdss_v2_20gb_curated/probes/redshift_probe_feature_ablation/runs/spectrum_only_z_spec/`
- `astro_sdss_v2_20gb_curated/probes/redshift_probe_feature_ablation/runs/both_z_img/`
- `astro_sdss_v2_20gb_curated/probes/redshift_probe_feature_ablation/runs/both_z_spec/`

### 12.4 本轮实验结果

新增实验结果：

| label | mode | input_key | best_epoch | MAE(z) | RMSE(z) | median |Δz|/(1+z) | outlier rate | Pearson | Spearman |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| image-only_z_img | image-only | z_img | 12 | 0.025318 | 0.033970 | 0.016165 | 0.000000 | 0.906747 | 0.787477 |
| spectrum-only_z_spec | spectrum-only | z_spec | 26 | 0.024465 | 0.033984 | 0.016505 | 0.000000 | 0.906672 | 0.811973 |
| both_z_img | both | z_img | 12 | 0.025318 | 0.033970 | 0.016165 | 0.000000 | 0.906747 | 0.787477 |
| both_z_spec | both | z_spec | 26 | 0.024465 | 0.033984 | 0.016505 | 0.000000 | 0.906672 | 0.811973 |

原 baseline（shared）结果：

| label | mode | input_key | best_epoch | MAE(z) | RMSE(z) | median |Δz|/(1+z) | outlier rate | Pearson | Spearman |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| image-only_z_shared | image-only | z_shared | 30 | 0.027434 | 0.037691 | 0.018217 | 0.000000 | 0.884902 | 0.705911 |
| spectrum-only_z_shared | spectrum-only | z_shared | 36 | 0.025436 | 0.035303 | 0.015998 | 0.000000 | 0.908785 | 0.681688 |
| both_z_shared | both | z_shared | 26 | 0.029204 | 0.038288 | 0.019223 | 0.000000 | 0.885878 | 0.695290 |

### 12.5 本轮结论

本轮最重要的结论是：

1. **`z_img` 明显优于 `image-only + z_shared`**
   - `MAE`: `0.025318 < 0.027434`
   - `RMSE`: `0.033970 < 0.037691`
   - 说明图像私有表征里确实有更强的 redshift 信息。

2. **`z_spec` 略优于 `spectrum-only + z_shared`**
   - `MAE`: `0.024465 < 0.025436`
   - `RMSE`: `0.033984 < 0.035303`
   - 说明 shared 表征没有完整保留光谱私有信息。

3. **`both + z_img` 与 `image-only + z_img` 基本一致**
   - 说明 `z_img` 主要由图像私有分支决定，没有因为联合输入明显增强。

4. **`both + z_spec` 与 `spectrum-only + z_spec` 基本一致**
   - 说明 `z_spec` 也主要由光谱私有分支决定，没有因为联合输入明显增强。

5. **最关键判断**
   - 当前问题不在于“图像分支完全没学到 redshift 信息”，因为 `z_img` 是有效的。
   - 当前真正的问题更像是：
     - **`z_shared` 没有充分吸收 private branch 中对 redshift 有用的信息**。
     - 也就是说，当前 shared representation 更偏“跨模态对齐”而不是“任务充分性”。

### 12.6 这轮实验后，下一步最推荐做什么

基于当前结果，后续优先级建议变为：

1. **做 `linear probe` vs `MLP probe` 对比**
   - 目标：判断 `z_shared / z_img / z_spec` 中的信息是否线性可读。

2. **做 `concat(z_shared, z_img)` / `concat(z_shared, z_spec)` / `concat(z_shared, z_img, z_spec)` 对比**
   - 目标：判断 shared 与 private 是否互补。

3. **做跨模态 retrieval / matching**
   - 目标：直接验证 unified representation 是否真的学到了跨模态对齐，而不仅仅是 redshift 回归。

当前不建议立刻回去大改 backbone；应先把 **“shared 缺了什么”** 这件事通过 probe 完全讲清楚。


---

## 13. 2026-03-09 新增实验：`linear probe vs MLP probe`

### 13.1 本轮实验目的

上一轮实验已经说明：
- `z_img` 与 `z_spec` 在 redshift probe 上优于 `z_shared`
- 当前 `z_shared` 没有充分吸收 private branch 中的 redshift 信息

这一轮实验进一步要回答：
- 这些表征中的 redshift 信息，是否是**线性可读**的？
- 还是必须依赖一个非线性 MLP 头才能读出来？

因此，本轮在现有 frozen feature 基础上新增了 `linear probe`，并与已有 `MLP probe` 做对比。

### 13.2 本轮新增代码改动

为了支持这轮实验，更新了：
- `scripts/train_astro_redshift_probe.py`

新增能力：
- 增加 `--head-type` 参数：
  - `mlp`
  - `linear`
- 在输出的 `metrics.json` 和 `best_probe_head.pt` 中记录 `head_type`

### 13.3 本轮运行命令

#### `image-only + z_shared + linear`

```bash
conda run -n mapanything-offline-310 python scripts/train_astro_redshift_probe.py \
  --features-root astro_sdss_v2_20gb_curated/probes/redshift_probe/features \
  --mode image-only \
  --output-dir astro_sdss_v2_20gb_curated/probes/redshift_probe_linear_ablation/image_only_z_shared_linear \
  --input-key z_shared \
  --head-type linear \
  --batch-size 64 \
  --epochs 100 \
  --patience 10 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --seed 42 \
  --device cuda
```

#### `image-only + z_img + linear`

```bash
conda run -n mapanything-offline-310 python scripts/train_astro_redshift_probe.py \
  --features-root astro_sdss_v2_20gb_curated/probes/redshift_probe/features \
  --mode image-only \
  --output-dir astro_sdss_v2_20gb_curated/probes/redshift_probe_linear_ablation/image_only_z_img_linear \
  --input-key z_img \
  --head-type linear \
  --batch-size 64 \
  --epochs 100 \
  --patience 10 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --seed 42 \
  --device cuda
```

#### `spectrum-only + z_shared + linear`

```bash
conda run -n mapanything-offline-310 python scripts/train_astro_redshift_probe.py \
  --features-root astro_sdss_v2_20gb_curated/probes/redshift_probe/features \
  --mode spectrum-only \
  --output-dir astro_sdss_v2_20gb_curated/probes/redshift_probe_linear_ablation/spectrum_only_z_shared_linear \
  --input-key z_shared \
  --head-type linear \
  --batch-size 64 \
  --epochs 100 \
  --patience 10 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --seed 42 \
  --device cuda
```

#### `spectrum-only + z_spec + linear`

```bash
conda run -n mapanything-offline-310 python scripts/train_astro_redshift_probe.py \
  --features-root astro_sdss_v2_20gb_curated/probes/redshift_probe/features \
  --mode spectrum-only \
  --output-dir astro_sdss_v2_20gb_curated/probes/redshift_probe_linear_ablation/spectrum_only_z_spec_linear \
  --input-key z_spec \
  --head-type linear \
  --batch-size 64 \
  --epochs 100 \
  --patience 10 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --seed 42 \
  --device cuda
```

#### `both + z_shared + linear`

```bash
conda run -n mapanything-offline-310 python scripts/train_astro_redshift_probe.py \
  --features-root astro_sdss_v2_20gb_curated/probes/redshift_probe/features \
  --mode both \
  --output-dir astro_sdss_v2_20gb_curated/probes/redshift_probe_linear_ablation/both_z_shared_linear \
  --input-key z_shared \
  --head-type linear \
  --batch-size 64 \
  --epochs 100 \
  --patience 10 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --seed 42 \
  --device cuda
```

### 13.4 本轮结果文件位置

结果目录：
- `astro_sdss_v2_20gb_curated/probes/redshift_probe_linear_ablation/`

线性 probe 汇总：
- `astro_sdss_v2_20gb_curated/probes/redshift_probe_linear_ablation/summary.csv`

与 MLP 合并对照：
- `astro_sdss_v2_20gb_curated/probes/redshift_probe_linear_ablation/combined_mlp_vs_linear.csv`

### 13.5 本轮实验结果

#### Linear probe 结果

| label | mode | input_key | head_type | best_epoch | MAE(z) | RMSE(z) | median |Δz|/(1+z) | outlier rate | Pearson | Spearman |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| image-only_z_shared_linear | image-only | z_shared | linear | 27 | 0.027375 | 0.036351 | 0.017976 | 0.000000 | 0.893173 | 0.732967 |
| image-only_z_img_linear | image-only | z_img | linear | 35 | 0.063670 | 0.091008 | 0.041472 | 0.072289 | 0.627187 | 0.372028 |
| spectrum-only_z_shared_linear | spectrum-only | z_shared | linear | 27 | 0.027529 | 0.036895 | 0.018426 | 0.000000 | 0.892039 | 0.719953 |
| spectrum-only_z_spec_linear | spectrum-only | z_spec | linear | 21 | 0.042311 | 0.053829 | 0.033923 | 0.000000 | 0.751784 | 0.617124 |
| both_z_shared_linear | both | z_shared | linear | 18 | 0.029761 | 0.038301 | 0.022351 | 0.000000 | 0.881080 | 0.706435 |

#### 与 MLP 的关键对照

| compare_key | linear MAE | mlp MAE | 结论 |
|---|---:|---:|---|
| image-only_z_shared | 0.027375 | 0.027434 | 基本相同，`z_shared` 几乎线性可读 |
| spectrum-only_z_shared | 0.027529 | 0.025436 | MLP 略优，但 linear 已经不错 |
| both_z_shared | 0.029761 | 0.029204 | 基本相同，`z_shared` 几乎线性可读 |
| image-only_z_img | 0.063670 | 0.025318 | 差距极大，`z_img` 依赖非线性头 |
| spectrum-only_z_spec | 0.042311 | 0.024465 | 差距极大，`z_spec` 依赖非线性头 |

### 13.6 本轮结论

这一轮的关键信息非常重要：

1. **`z_shared` 几乎是线性可读的**
   - 无论 `image-only` 还是 `both`，`linear` 与 `MLP` 差距都很小。
   - 说明 `z_shared` 中的 redshift 信息已经被组织成比较规则、低复杂度的形式。

2. **`z_img` 与 `z_spec` 明显依赖非线性头**
   - `z_img`: `linear` 远差于 `MLP`
   - `z_spec`: `linear` 也远差于 `MLP`
   - 说明 private representation 中虽然含有更强信息，但这些信息分布更复杂，不是简单线性可读的。

3. **综合解释**
   - `z_shared`：信息更“规整”、更线性、但信息量不够满
   - `z_img / z_spec`：信息量更强，但更“私有”、更非线性、可读性依赖更强头部

4. **这对 backbone 的解释非常关键**
   - 当前 shared branch 的优势不是“信息最强”，而是“结构最规整”
   - 当前 private branch 的优势不是“更统一”，而是“保留了更丰富的任务相关细节”

### 13.7 这轮实验后的最优先下一步

基于当前两轮 probe 结果，最推荐的下一步已经非常明确：

1. **做 `concat` 实验**
   - `concat(z_shared, z_img)`
   - `concat(z_shared, z_spec)`
   - `concat(z_shared, z_img, z_spec)`

目标：
- 判断 shared 与 private 是否互补
- 判断 shared 是否真的缺少 private 的关键信息

2. **做跨模态 retrieval / matching**

目标：
- 直接验证 `z_shared` 是否真的学到了跨模态对齐
- 这样就能把“shared 更规整但不一定任务最强”这个结论，转成更完整的 foundation 表征论证

当前暂时**仍然不建议马上大改 backbone**；先把 probe 证据链补完整，收益更大。


---

## 14. 2026-03-09 新增实验：`concat` probe（shared 与 private 是否互补）

### 14.1 本轮实验目的

前两轮实验已经得到两个关键结论：
- `z_shared` 更规整、几乎线性可读
- `z_img / z_spec` 信息更强，但更依赖非线性任务头

因此本轮实验要回答的核心问题是：
- `shared` 和 `private` 是不是互补？
- 如果把它们拼接起来，是否能比单独使用更好？

### 14.2 本轮新增代码改动

更新文件：
- `scripts/train_astro_redshift_probe.py`

新增能力：
- `--input-key` 现在支持逗号分隔的多个特征键
- 多个特征会在特征维度直接拼接后再送入 probe head

例如：
- `--input-key z_shared,z_img`
- `--input-key z_shared,z_spec`
- `--input-key z_shared,z_img,z_spec`

### 14.3 本轮运行命令

#### `image-only + concat(z_shared, z_img)`

```bash
conda run -n mapanything-offline-310 bash -lc 'CUDA_VISIBLE_DEVICES=0 python scripts/train_astro_redshift_probe.py \
  --features-root astro_sdss_v2_20gb_curated/probes/redshift_probe/features \
  --mode image-only \
  --output-dir astro_sdss_v2_20gb_curated/probes/redshift_probe_concat_ablation/image_only_z_shared_z_img \
  --input-key z_shared,z_img \
  --head-type mlp \
  --batch-size 64 \
  --epochs 100 \
  --patience 10 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --dropout 0.1 \
  --hidden-dims 256,64 \
  --seed 42 \
  --device cuda'
```

#### `spectrum-only + concat(z_shared, z_spec)`

```bash
conda run -n mapanything-offline-310 bash -lc 'CUDA_VISIBLE_DEVICES=1 python scripts/train_astro_redshift_probe.py \
  --features-root astro_sdss_v2_20gb_curated/probes/redshift_probe/features \
  --mode spectrum-only \
  --output-dir astro_sdss_v2_20gb_curated/probes/redshift_probe_concat_ablation/spectrum_only_z_shared_z_spec \
  --input-key z_shared,z_spec \
  --head-type mlp \
  --batch-size 64 \
  --epochs 100 \
  --patience 10 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --dropout 0.1 \
  --hidden-dims 256,64 \
  --seed 42 \
  --device cuda'
```

#### `both + concat(z_shared, z_img)`

```bash
conda run -n mapanything-offline-310 bash -lc 'CUDA_VISIBLE_DEVICES=2 python scripts/train_astro_redshift_probe.py \
  --features-root astro_sdss_v2_20gb_curated/probes/redshift_probe/features \
  --mode both \
  --output-dir astro_sdss_v2_20gb_curated/probes/redshift_probe_concat_ablation/both_z_shared_z_img \
  --input-key z_shared,z_img \
  --head-type mlp \
  --batch-size 64 \
  --epochs 100 \
  --patience 10 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --dropout 0.1 \
  --hidden-dims 256,64 \
  --seed 42 \
  --device cuda'
```

#### `both + concat(z_shared, z_spec)`

```bash
conda run -n mapanything-offline-310 bash -lc 'CUDA_VISIBLE_DEVICES=3 python scripts/train_astro_redshift_probe.py \
  --features-root astro_sdss_v2_20gb_curated/probes/redshift_probe/features \
  --mode both \
  --output-dir astro_sdss_v2_20gb_curated/probes/redshift_probe_concat_ablation/both_z_shared_z_spec \
  --input-key z_shared,z_spec \
  --head-type mlp \
  --batch-size 64 \
  --epochs 100 \
  --patience 10 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --dropout 0.1 \
  --hidden-dims 256,64 \
  --seed 42 \
  --device cuda'
```

#### `both + concat(z_shared, z_img, z_spec)`

```bash
conda run -n mapanything-offline-310 bash -lc 'CUDA_VISIBLE_DEVICES=4 python scripts/train_astro_redshift_probe.py \
  --features-root astro_sdss_v2_20gb_curated/probes/redshift_probe/features \
  --mode both \
  --output-dir astro_sdss_v2_20gb_curated/probes/redshift_probe_concat_ablation/both_z_shared_z_img_z_spec \
  --input-key z_shared,z_img,z_spec \
  --head-type mlp \
  --batch-size 64 \
  --epochs 100 \
  --patience 10 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --dropout 0.1 \
  --hidden-dims 256,64 \
  --seed 42 \
  --device cuda'
```

### 14.4 本轮结果文件位置

结果目录：
- `astro_sdss_v2_20gb_curated/probes/redshift_probe_concat_ablation/`

本轮汇总：
- `astro_sdss_v2_20gb_curated/probes/redshift_probe_concat_ablation/summary.csv`

与 baseline / single-feature 合并结果：
- `astro_sdss_v2_20gb_curated/probes/redshift_probe_concat_ablation/combined_with_baselines.csv`

### 14.5 本轮实验结果

#### concat probe 结果

| label | mode | input_key | MAE(z) | RMSE(z) | median |Δz|/(1+z) | outlier rate | Pearson | Spearman |
|---|---|---|---:|---:|---:|---:|---:|---:|
| image-only_z_shared_z_img | image-only | z_shared,z_img | 0.027179 | 0.035530 | 0.018184 | 0.000000 | 0.903403 | 0.772365 |
| spectrum-only_z_shared_z_spec | spectrum-only | z_shared,z_spec | 0.022446 | 0.031437 | 0.012772 | 0.000000 | 0.922038 | 0.789136 |
| both_z_shared_z_img | both | z_shared,z_img | 0.027405 | 0.037584 | 0.017423 | 0.000000 | 0.886186 | 0.740061 |
| both_z_shared_z_spec | both | z_shared,z_spec | 0.023899 | 0.033205 | 0.013921 | 0.000000 | 0.911425 | 0.772785 |
| both_z_shared_z_img_z_spec | both | z_shared,z_img,z_spec | 0.028075 | 0.036258 | 0.018433 | 0.000000 | 0.901648 | 0.760254 |

#### 与之前结果的关键对照

| compare | MAE(z) |
|---|---:|
| image-only_z_shared | 0.027434 |
| image-only_z_img | 0.025318 |
| image-only_z_shared_z_img | 0.027179 |
| spectrum-only_z_shared | 0.025436 |
| spectrum-only_z_spec | 0.024465 |
| spectrum-only_z_shared_z_spec | 0.022446 |
| both_z_shared | 0.029204 |
| both_z_img | 0.025318 |
| both_z_spec | 0.024465 |
| both_z_shared_z_img | 0.027405 |
| both_z_shared_z_spec | 0.023899 |
| both_z_shared_z_img_z_spec | 0.028075 |

### 14.6 本轮结论

1. **`shared + spec` 是当前最有效的组合**
   - `spectrum-only`: `z_shared + z_spec = 0.022446`，优于 `z_shared` 和 `z_spec` 单独使用
   - `both`: `z_shared + z_spec = 0.023899`，同样优于 `z_shared`、`z_img`、`z_spec` 单独使用

2. **`shared + img` 没有明显收益**
   - `image-only`: `z_shared + z_img = 0.027179`，只比 `z_shared` 略好，但不如 `z_img` 单独
   - `both`: `z_shared + z_img = 0.027405`，明显不如 `z_spec` 或 `shared + spec`

3. **`shared + img + spec` 也不如 `shared + spec`**
   - `both_z_shared_z_img_z_spec = 0.028075`
   - 说明把 `z_img` 也加进来，并没有进一步提升，反而引入了噪声或冗余

4. **当前最准确的结构性结论**
   - `z_shared` 和 `z_spec` 是互补的
   - `z_shared` 和 `z_img` 互补性弱
   - 在 redshift 这个任务上，当前 backbone 的有效信息主轴仍然是：
     - **`spectrum private information` + `shared structured information`**

5. **更进一步的解释**
   - `z_shared` 提供的是规整、可读、跨模态一致的语义结构
   - `z_spec` 提供的是最关键的任务细节
   - `z_img` 当前没有在 redshift 上形成额外稳定增益

### 14.7 这轮实验后的最优先下一步

基于现在三轮 probe 的完整证据链，下一步最值得做的是：

1. **跨模态 retrieval / matching**
   - 用 `z_shared` 做 image ↔ spectrum 检索
   - 这是当前最能证明 unified representation 真的学到了跨模态对齐的实验

2. **如果还继续 redshift 方向**
   - 可以把 `z_shared + z_spec` 当作当前最强下游输入组合
   - 用它做更正式的 photo-z 结果展示

3. **如果目标是 foundation model 叙事**
   - 当前故事已经比较完整：
     - `z_shared` 更规整、更线性
     - `z_spec` 含有更强任务细节
     - `z_shared + z_spec` 最优，说明 unified 与 modality-private 是互补而不是替代关系


---

## 15. 2026-03-09 新增实验：cross-modal retrieval / matching

### 15.1 本轮实验目的

前面几轮 redshift probe 已经回答了：
- `z_shared` 更规整、更线性可读
- `z_spec` 承载更多任务细节
- `z_shared + z_spec` 对 redshift 最有效

但这仍然主要是在回答“能不能做 redshift”。

为了验证 unified representation 是否真的学到了 **跨模态对齐能力**，本轮新增了：
- **image → spectrum retrieval**
- **spectrum → image retrieval**

核心目标是：
- 给定一个图像 embedding，能否在光谱库中找回正确的配对样本？
- 给定一个光谱 embedding，能否在图像库中找回正确的配对样本？

### 15.2 本轮新增代码改动

新增脚本：
- `scripts/eval_astro_cross_modal_retrieval.py`
- `scripts/run_astro_cross_modal_retrieval.py`

#### `scripts/eval_astro_cross_modal_retrieval.py`
作用：
- 读取 frozen feature `.npz`
- 构建 query / gallery embedding
- 计算相似度矩阵（默认 cosine）
- 评估 retrieval 指标：
  - `Recall@1`
  - `Recall@5`
  - `Recall@10`
  - `MRR`
  - `mean_rank`
  - `median_rank`
  - `similarity_auc`
- 保存：
  - `metrics.json`
  - `per_query.csv`
  - `similarity_matrix.npz`
  - `similarity_heatmap.png`
  - `similarity_hist.png`
  - `rank_hist.png`

#### `scripts/run_astro_cross_modal_retrieval.py`
作用：
- 一键跑标准 retrieval 配置
- 自动汇总为 `summary.csv`

### 15.3 本轮运行命令

#### 单个 retrieval 评估

例如：`image-only z_shared -> spectrum-only z_shared`

```bash
conda run -n mapanything-offline-310 python scripts/eval_astro_cross_modal_retrieval.py \
  --features-root astro_sdss_v2_20gb_curated/probes/redshift_probe/features \
  --split val \
  --query-mode image-only \
  --gallery-mode spectrum-only \
  --query-key z_shared \
  --gallery-key z_shared \
  --output-dir astro_sdss_v2_20gb_curated/probes/cross_modal_retrieval/image_zshared_to_spec_zshared
```

#### 一键跑标准 retrieval 集合

```bash
conda run -n mapanything-offline-310 python scripts/run_astro_cross_modal_retrieval.py \
  --features-root astro_sdss_v2_20gb_curated/probes/redshift_probe/features \
  --output-root astro_sdss_v2_20gb_curated/probes/cross_modal_retrieval \
  --split val
```

### 15.4 本轮结果文件位置

根目录：
- `astro_sdss_v2_20gb_curated/probes/cross_modal_retrieval/`

汇总文件：
- `astro_sdss_v2_20gb_curated/probes/cross_modal_retrieval/summary.csv`

每个 retrieval 配置目录下都有：
- `metrics.json`
- `per_query.csv`
- `similarity_matrix.npz`
- `similarity_heatmap.png`
- `similarity_hist.png`
- `rank_hist.png`

### 15.5 本轮实验结果

本轮在 `val=83` 对配样本上评估，结果如下：

| label | query -> gallery | feature | Recall@1 | Recall@5 | Recall@10 | MRR | mean rank | median rank | AUC |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| image_zshared_to_spec_zshared | image -> spectrum | z_shared | 0.1928 | 0.5301 | 0.6988 | 0.3495 | 10.23 | 5 | 0.8776 |
| spec_zshared_to_image_zshared | spectrum -> image | z_shared | 0.2169 | 0.5422 | 0.7711 | 0.3883 | 8.16 | 4 | 0.8776 |
| image_lshared_to_spec_lshared | image -> spectrum | latent_shared_proj | 0.1807 | 0.5181 | 0.6747 | 0.3375 | 10.43 | 5 | 0.8786 |
| spec_lshared_to_image_lshared | spectrum -> image | latent_shared_proj | 0.2048 | 0.5542 | 0.7590 | 0.3802 | 8.35 | 4 | 0.8786 |
| image_zimg_to_spec_zspec | image -> spectrum | z_img vs z_spec | 0.2048 | 0.5060 | 0.6627 | 0.3495 | 12.04 | 5 | 0.8581 |
| spec_zspec_to_image_zimg | spectrum -> image | z_spec vs z_img | 0.1687 | 0.5422 | 0.6506 | 0.3428 | 12.34 | 4 | 0.8581 |

### 15.6 本轮结论

1. **`z_shared` 的 cross-modal retrieval 是有效的，而且明显高于随机**
   - `83` 个候选里随机 `Recall@1` 约等于 `1/83 ≈ 0.012`
   - 当前 `z_shared` 的 `Recall@1 ≈ 0.19 ~ 0.22`
   - 这已经说明 unified representation 确实学到了跨模态配对关系

2. **`latent_shared_proj` 和 `z_shared` 的 retrieval 表现接近**
   - projected alignment space 没有显著压倒 `z_shared`
   - 说明当前 shared 主表征本身就带有较强的跨模态对齐能力

3. **`z_img ↔ z_spec` 也能做 retrieval，但整体不如 shared space 稳定**
   - `Recall@1` 有时接近，但 `mean rank` 更差，`AUC` 也更低
   - 说明 private spaces 之间不是最自然的统一对齐空间

4. **这轮实验和前面几轮 probe 拼起来，故事已经非常清楚**
   - `z_shared`：
     - 线性可读
     - 结构规整
     - 跨模态检索有效
   - `z_spec`：
     - 对 redshift 更强
     - 含任务细节更多
   - `z_shared + z_spec`：
     - 最适合作为 redshift 下游输入

5. **因此当前最合理的论文叙事是**
   - `z_shared` 是统一天文表征主轴，负责跨模态对齐与语义结构化
   - `z_spec` / `z_img` 是 modality-private 细节补充
   - 对具体科学任务，下游可以将 unified 与 private 信息联合使用

### 15.7 当前阶段最重要的总判断

到这里，关于 `v2 final` 的核心问题已经基本回答完毕：

- 它**不是**一个空 backbone
- 它**确实学到了统一天文表征**
- 这个统一表征 **能支持跨模态 retrieval**
- 同时在具体任务上，`shared` 与 `private` 是 **互补关系**，不是简单替代关系

### 15.8 现在下一步最推荐做什么

当前最推荐的下一步有两个方向：

#### 方向 A：整理成论文级展示材料

把现有结果整理成统一图表：
- redshift probe 对比表
- linear vs MLP 对比表
- shared / private / concat 对比表
- cross-modal retrieval 结果表
- 检索可视化图

#### 方向 B：开始做正式下游任务头

既然已经证明 backbone 有用，就可以开始更正式的下游：
- `photo-z` 正式模型（可用 `z_shared + z_spec`）
- 分类 / anomaly / retrieval-based ranking

如果目标是“尽快做出能说服编委的结果”，建议优先走：
- **整理展示材料 + 一个正式 photo-z 头**

