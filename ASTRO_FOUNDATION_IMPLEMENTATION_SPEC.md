# Astronomy Foundation Encoder 实现规格书

版本：v1.0  
日期：2026-03-09  
状态：可进入实现阶段

---

## 1. 文档目标

本文档将“统一天文表征基础模型”的设计，细化为可直接实现的工程规格。

本文档解决的问题不是“做什么方向”，而是：
- 要新增哪些模块
- 每个模块输入输出是什么
- 训练流程如何拆分
- loss 如何组织
- 配置和脚本如何命名
- 评估矩阵如何固化
- 如何从现有 `v2` 仓库迁移到新的解耦式框架

该文档默认以当前仓库为实现载体：
- 当前 backbone 与实验结论记录：`ASTRO_PROGRESS_LOG.md`
- 当前 `v2` 模型入口：`configs/model/astro_mapanything_v2.yaml`
- 当前 `v2` 数据入口：`configs/dataset/astro_sdss_pair_v2.yaml`
- 当前 `v2` loss 入口：`configs/loss/astro_bidirectional_v2.yaml`

---

## 2. 项目目标与非目标

### 2.1 项目目标

实现一个新的 **Astronomy Foundation Encoder**，其核心要求是：

1. 输入天文多模态数据：
   - 5-band 图像
   - 1D 光谱
   - 可选 metadata
2. 输出统一表征与私有表征：
   - `z_shared`
   - `z_img_private`
   - `z_spec_private`
   - `z_nuisance`
3. 表征训练与下游训练彻底解耦
4. backbone 训练目标以表征学习为主，而不是具体任务头
5. 下游任务通过标准化 probe / finetune 方式接入

### 2.2 非目标

以下内容不作为 backbone 主目标：

1. 不以 `spectrum -> image` 高保真生成作为 backbone 主任务
2. 不把 5-band 图像强行等价为 RGB 图像问题
3. 不把下游 `photo-z` 监督作为 backbone 主监督
4. 不删除 private branch
5. 不将 metadata 无条件注入所有下游任务

---

## 3. 现有实验结论作为设计约束

新系统必须满足并利用以下已验证事实：

1. `z_shared` 有效，可做 retrieval，可做 probe
2. `z_shared` 更规整、更线性可读
3. `z_spec` 对 redshift 更强
4. `z_img` 有信息，但对 redshift 不是最稳定补益来源
5. `z_shared + z_spec` 是当前最优 redshift 输入组合
6. `shared` 与 `private` 是互补关系，不是替代关系
7. `z_shared` 的跨模态检索显著优于随机

这些结论意味着最终系统必须保留：
- shared branch
- private branch
- retrieval-focused evaluation
- nuisance separation

---

## 4. 顶层代码结构

建议在当前仓库中新增如下结构：

```text
mapanything/
  models/
    astro_foundation/
      __init__.py
      types.py
      image_frontend.py
      spectrum_frontend.py
      shared_trunk.py
      heads.py
      model.py
  train/
    losses_astro_foundation.py
  utils/
    astro_foundation_inference.py

configs/
  model/
    astro_foundation_encoder.yaml
  dataset/
    astro_sdss_foundation.yaml
  loss/
    astro_foundation.yaml
  train_params/
    astro_foundation_pretrain.yaml
    astro_foundation_probe.yaml
    astro_foundation_finetune.yaml
  train_astro_foundation.yaml
  train_astro_photoz_head.yaml

scripts/
  train_astro_foundation.py
  export_astro_foundation_features.py
  eval_astro_probe_suite.py
  eval_astro_cross_modal_retrieval.py
  run_astro_cross_modal_retrieval.py
  train_astro_photoz_head.py
  compare_astro_foundation_checkpoints.py

notebooks/
  astro_foundation_probe_suite.ipynb
  astro_foundation_retrieval_demo.ipynb
  astro_foundation_case_studies.ipynb
```

当前已有可复用脚本：
- `scripts/export_astro_probe_features.py`
- `scripts/train_astro_redshift_probe.py`
- `scripts/eval_astro_cross_modal_retrieval.py`
- `scripts/run_astro_cross_modal_retrieval.py`

新实现应尽量复用这些工具链，避免重复造轮子。

---

## 5. 数据接口规格

## 5.1 输入模态

### 图像输入
- 名称：`img_astro`
- 形状：`[B, 5, H, W]`
- 默认尺寸：`H=W=224`
- band 顺序：`u,g,r,i,z`

### 光谱输入
- 名称：`spec_astro`
- 形状：`[B, C_spec, L]`
- 默认：`C_spec=4, L=2048`
- 当前建议保留四通道输入，但在最终版要明确语义化拆分：
  - flux / normalized flux
  - rest-frame / observed-frame variant
  - residual-like channel
  - ivar-weighted or uncertainty-related channel

### Metadata 输入
- 名称：`meta_cond`
- 形状：`[B, D_meta]`
- 默认：`D_meta=4`
- 当前仓库中为：`ra, dec, log1p(z), sn_median_r`

### 严格约束
- 对任何 redshift 下游任务：**metadata 默认禁用**
- 新框架需支持 safe metadata 与 leakage-prone metadata 分离

---

## 5.2 数据集返回结构

foundation 预训练阶段的数据集返回必须统一为两个 view：

```python
[
  view_image: {
    "img_astro": Tensor,
    "img_masked_astro": Tensor | None,
    "img_valid_mask": BoolTensor,
    "img_mask_tokens": BoolTensor,
    "img_band_mask": BoolTensor | None,
    "image_input_mask": BoolTensor,
    "meta_cond": Tensor,
    "meta_valid": BoolTensor,
    "sample_id": LongTensor,
    "epoch_idx": LongTensor,
  },
  view_spectrum: {
    "spec_astro": Tensor,
    "spec_masked_astro": Tensor | None,
    "spec_valid_mask": BoolTensor,
    "spec_mask_tokens": BoolTensor,
    "spec_ivar_weight": Tensor,
    "spectrum_input_mask": BoolTensor,
    "meta_cond": Tensor,
    "meta_valid": BoolTensor,
    "sample_id": LongTensor,
    "epoch_idx": LongTensor,
  }
]
```

该接口与当前 `AstroSDSSPairDatasetV2` 保持兼容方向，但新数据集类需要更明确命名与字段语义。

建议新增：
- `AstroSDSSFoundationDataset`

文件位置：
- `mapanything/datasets/astro/sdss_foundation.py`

---

## 6. 模型模块规格

## 6.1 `types.py`

新增统一输出 dataclass：

```python
@dataclass
class AstroFoundationOutput:
    z_shared: torch.Tensor
    z_img_private: torch.Tensor
    z_spec_private: torch.Tensor
    z_nuisance: torch.Tensor
    latent_shared_proj: torch.Tensor
    latent_img_proj: torch.Tensor
    latent_spec_proj: torch.Tensor
    pred_nuisance: torch.Tensor | None
    pred_match_logit: torch.Tensor | None
    pred_img_masked: torch.Tensor | None
    pred_spec_masked: torch.Tensor | None
```

目标：统一 backbone 与 inference / probe / finetune 的接口。

---

## 6.2 `image_frontend.py`

### 类名
- `BandAwarePatchEmbed`
- `AstroImageFrontend`

### 目标
在 ViT/DINO 风格图像前端中显式保留 band identity，不把 5-band 过早压成“普通 3 通道视觉特征”。

### 输入
- `img_astro: [B, 5, H, W]`
- `img_valid_mask: [B, H, W]` 可选
- `img_band_mask: [B, 5]` 可选

### 输出
- `patch_tokens: [B, N_img, D]`
- `register_tokens: [B, N_reg, D]`
- `band_summary_tokens: [B, 5, D]` 可选

### 必要实现点
1. 支持 native 5-channel patch embedding
2. 加入 band embedding
3. 支持 register tokens
4. 支持输入 mask / band mask
5. 保留从 DINOv2 初始化的路径，但不要求完全绑定 DINOv2

### 实现建议
- 首版可直接在当前 `astro_mapanything` 图像编码器基础上抽象重构
- register tokens 初版数量建议：`N_reg = 4 or 8`

---

## 6.3 `spectrum_frontend.py`

### 类名
- `AstroSpectrumPatchify`
- `AstroSpectrumFrontend`

### 目标
为 1D 光谱建立强表达前端，而不是只做简单序列压缩。

### 输入
- `spec_astro: [B, C_spec, L]`
- `spec_valid_mask: [B, L]`
- `spec_mask_tokens: [B, L]`
- `spec_ivar_weight: [B, L]`

### 输出
- `spectrum_tokens: [B, N_spec, D]`
- `spectrum_global_token: [B, D]` 可选

### 必要实现点
1. 1D patch / span tokenization
2. wavelength positional encoding
3. ivar-aware token weighting
4. masked span support
5. 强化 line-sensitive representation

### 首版实现建议
- 采用 1D Transformer encoder
- patch stride 可与当前 `spec_stride` 兼容
- hidden dim 与 image branch 对齐：`D=768`

---

## 6.4 `shared_trunk.py`

### 类名
- `SharedLatentBottleneck`
- `CrossModalSharedTrunk`

### 目标
构建 unified representation 的主干，不使用简单 token concat 直接替代。

### 输入
- `image_tokens: [B, N_img, D]`
- `spectrum_tokens: [B, N_spec, D]`
- `meta_tokens: [B, N_meta, D] | None`
- `image_register_tokens: [B, N_reg, D] | None`

### 输出
- `shared_tokens: [B, N_shared, D]`
- `updated_image_tokens: [B, N_img, D]`
- `updated_spectrum_tokens: [B, N_spec, D]`
- `updated_nuisance_tokens: [B, N_reg, D] | None`

### 必要实现点
1. learnable shared latent tokens
2. image/spec → shared cross-attention
3. shared self-attention refinement
4. optional shared → modality feedback
5. missing modality support

### 默认超参建议
- `N_shared = 8`
- `D = 768`
- `num_layers = 8 ~ 12`
- `num_heads = 12`

---

## 6.5 `heads.py`

### 类名
- `SharedPoolingHead`
- `PrivatePoolingHead`
- `NuisanceHead`
- `MatchingHead`
- `MaskedSpectrumDecoder`
- `MaskedImageDecoder`（可选）

### 目标
从 trunk 输出中显式得到四类表征与训练辅助头。

### 输出定义
- `z_shared`
- `z_img_private`
- `z_spec_private`
- `z_nuisance`
- `latent_shared_proj`
- `latent_img_proj`
- `latent_spec_proj`
- `pred_match_logit`
- `pred_nuisance`
- `pred_img_masked`
- `pred_spec_masked`

### 注意
- `MaskedImageDecoder` 只能作为辅助组件，不允许成为主容量吞噬者
- `pred_match_logit` 是 backbone 训练期辅助头，不是最终下游头

---

## 6.6 `model.py`

### 类名
- `AstroFoundationEncoder`

### 核心方法

#### `forward_pretrain(...)`
用于 backbone 训练：
- 返回 `AstroFoundationOutput`
- 含所有训练期辅助输出

#### `encode(...)`
用于推理与导出：
- 只返回表征相关字段
- 与当前 `extract_backbone_features` 兼容方向保持一致

#### `forward_infer(...)`
用于 retrieval / probe 导出

### 输入接口
沿用当前 `views = [view_image, view_spectrum]` 风格，避免打断现有工具链。

### 输出字段必须固定
最少包括：
- `z_shared`
- `z_img_private`
- `z_spec_private`
- `z_nuisance`
- `latent_shared_proj`
- `latent_img_proj`
- `latent_spec_proj`

---

## 7. Loss 规格

新增文件：
- `mapanything/train/losses_astro_foundation.py`

新增主类：
- `AstroFoundationLoss`

---

## 7.1 Loss 总结构

```text
L_total =
  w_align * L_align
+ w_match * L_match
+ w_img_mask * L_img_mask
+ w_spec_mask * L_spec_mask
+ w_consistency * L_consistency
+ w_disentangle * L_disentangle
+ w_nuisance * L_nuisance
+ w_aux_science * L_aux_science
```

---

## 7.2 `L_align`

### 目的
同一对象的 image / spectrum shared embedding 接近，非配对样本远离。

### 实现
- symmetric InfoNCE
- in-batch negatives
- optional hard-negative mining

### 输入
- `latent_shared_proj_from_image`
- `latent_shared_proj_from_spectrum`

---

## 7.3 `L_match`

### 目的
训练一个明确的配对判别能力。

### 实现
- binary classification on matched / unmatched pairs
- 正样本：同 `sample_id`
- 负样本：in-batch mismatch + hard negative

---

## 7.4 `L_img_mask`

### 目的
图像模态内表征稳定性

### 实现
- masked patch prediction
- 只做轻量 patch-level reconstruction / feature regression
- 不做高保真自然图像生成目标

---

## 7.5 `L_spec_mask`

### 目的
光谱模态内鲁棒性

### 实现
- masked span regression
- ivar-weighted robust regression
- line-region optional emphasis

---

## 7.6 `L_consistency`

### 目的
同对象在三种输入状态下 shared 表征一致：
- image-only
- spectrum-only
- both

### 实现
- cosine consistency / L2 consistency
- 可在 `z_shared` 或 `latent_shared_proj` 上实施

---

## 7.7 `L_disentangle`

### 目的
减少 shared / private / nuisance 混叠。

### 实现建议
首版采用：
- cross-covariance decorrelation
- orthogonality regularization

不建议首版就上复杂 adversarial MI 估计器。

---

## 7.8 `L_nuisance`

### 目的
把背景噪声吸收到 `z_nuisance`

### 可监督目标
- quality ratio
- mask ratio
- sky background summary
- noise strength proxy
- artifact bucket

### 首版建议
使用简单的 regression / classification multi-head。

---

## 7.9 `L_aux_science`

### 目的
轻量科学结构约束

### 可选任务
- coarse redshift bin
- line presence
- morphology coarse bucket
- S/N bucket

### 约束
- 默认权重小
- 不允许主导 backbone 训练方向

---

## 8. 训练阶段与配置规格

新增主训练配置：
- `configs/train_astro_foundation.yaml`

### 8.1 Stage 0: setup
- stats / normalization
- mask / ivar / quality pipeline 确认

### 8.2 Stage 1: unimodal warmup
- image masked modeling
- spectrum masked modeling
- nuisance learning
- 不强调强对齐

### 8.3 Stage 2: shared alignment pretraining
- both 输入占主导
- contrastive + matching + consistency

### 8.4 Stage 3: missing-modality robustness
- both / image-only / spectrum-only 混合
- 强化 consistency 与 shared stability

### 8.5 Stage 4: retrieval sharpening
- hard negatives
- matching emphasis
- retrieval 指标进验证与 checkpoint 选择

---

## 9. 配置文件规格

## 9.1 `configs/model/astro_foundation_encoder.yaml`
应包含：
- image frontend settings
- spectrum frontend settings
- shared trunk settings
- nuisance token settings
- projection dims
- metadata usage flags

## 9.2 `configs/loss/astro_foundation.yaml`
应包含：
- 各损失权重
- hard negative 开关
- masked modeling 配置
- consistency 目标配置

## 9.3 `configs/dataset/astro_sdss_foundation.yaml`
应包含：
- train/val manifests
- image/spectrum masking 参数
- stage-specific modality probabilities
- metadata safety options

## 9.4 `configs/train_params/astro_foundation_pretrain.yaml`
应包含：
- batch size
- epochs
- warmup
- LR schedule
- AMP
- save/eval freq

## 9.5 `configs/train_astro_photoz_head.yaml`
应包含：
- feature root
- input key combination
- head type
- freeze / finetune policy
- evaluation metrics

---

## 10. 脚本规格

## 10.1 `scripts/train_astro_foundation.py`
- backbone 主训练入口
- Hydra 驱动
- 支持 stage-aware logging
- checkpoint 保存为：
  - `checkpoint-best-retrieval.pth`
  - `checkpoint-best-probe.pth`
  - `checkpoint-final.pth`

## 10.2 `scripts/export_astro_foundation_features.py`
- 导出：
  - `z_shared`
  - `z_img_private`
  - `z_spec_private`
  - `z_nuisance`
  - projection features
- 支持：
  - image-only
  - spectrum-only
  - both

## 10.3 `scripts/eval_astro_probe_suite.py`
- 一键跑：
  - linear probe
  - mlp probe
  - concat probe

## 10.4 `scripts/train_astro_photoz_head.py`
- 正式 photo-z 头训练
- 默认支持输入：
  - `z_shared + z_spec_private`

## 10.5 `scripts/eval_astro_cross_modal_retrieval.py`
- 已实现，可继续复用
- 后续只需扩展更多特征组合

---

## 11. 标准评估套件

## 11.1 Probe Suite
对每个特征组合默认跑：
- linear
- mlp

默认组合：
- `z_shared`
- `z_img_private`
- `z_spec_private`
- `z_shared + z_img_private`
- `z_shared + z_spec_private`
- `z_shared + z_img_private + z_spec_private`

默认任务：
- redshift regression

---

## 11.2 Retrieval Suite
默认跑：
- image → spectrum
- spectrum → image

默认特征：
- `z_shared`
- `latent_shared_proj`
- `z_img_private ↔ z_spec_private`

---

## 11.3 正式任务套件
至少包含：
- photo-z
- classification
- retrieval ranking / matching

---

## 12. 输出目录规范

建议统一放在：

```text
astro_sdss_v2_20gb_curated/
  foundation/
    experiments/
    features/
    probes/
    retrieval/
    tasks/
```

### 推荐子目录
- backbone 训练：`foundation/experiments/`
- feature 导出：`foundation/features/`
- probe：`foundation/probes/`
- retrieval：`foundation/retrieval/`
- photo-z 任务头：`foundation/tasks/photoz/`

---

## 13. 迁移策略

## 13.1 不删除现有 v2 工具链
保留以下现有资源作为 baseline：
- 当前 `v2 final checkpoint`
- 当前 probe 结果
- 当前 retrieval 结果
- 当前 scripts 工具链

## 13.2 迁移顺序

### 第一步
新增新模型目录与类型定义：
- `astro_foundation/types.py`
- `astro_foundation/model.py`

### 第二步
迁移并重构前端：
- image frontend
- spectrum frontend

### 第三步
实现 shared trunk 与 heads

### 第四步
实现新的 loss 与训练脚本

### 第五步
接上 feature export / probe / retrieval

### 第六步
开始 backbone 预训练

---

## 14. 验收标准

### 第一层：工程验收
- backbone 可训练
- feature 可导出
- probe / retrieval 脚本可跑通

### 第二层：表示验收
- `z_shared` 可做线性 probe
- `z_shared` retrieval 明显优于随机

### 第三层：任务验收
- `z_shared + z_spec_private` 的 photo-z 正式头优于当前 probe baseline

### 第四层：科学叙事验收
需要能清楚说明：
- shared 学了什么
- private 学了什么
- nuisance 学了什么
- 为什么 shared + private 是最佳接口

---

## 15. 实施顺序（推荐）

### Phase 1：框架搭建
1. 新建 `astro_foundation` 模块骨架
2. 定义统一输出 dataclass
3. 重构 image / spectrum frontend
4. 实现 shared trunk
5. 实现 heads

### Phase 2：训练系统
6. 实现 `losses_astro_foundation.py`
7. 新建 Hydra configs
8. 新建 `train_astro_foundation.py`

### Phase 3：评估系统
9. 新建 `export_astro_foundation_features.py`
10. 复用 / 迁移 probe 脚本
11. 复用 retrieval 脚本
12. 新建 `eval_astro_probe_suite.py`

### Phase 4：正式任务
13. 新建 `train_astro_photoz_head.py`
14. 新建 classification / ranking 任务头

---

## 16. 当前阶段建议

在正式动手重构前，推荐默认把以下内容视为新系统的硬约束：

1. backbone 训练与下游训练彻底解耦
2. `z_shared` / `z_img_private` / `z_spec_private` / `z_nuisance` 四表征并存
3. retrieval 是 backbone 评估的一级指标，不再只是附属图
4. `photo-z` 正式头默认输入组合为：
   - `z_shared + z_spec_private`
5. `s2i` 生成不进入 backbone 主目标

---

## 17. 结论

该规格书对应的实现目标是：
- 构建一个真正面向天文多模态 foundation representation 的 encoder
- 让 unified representation、private information、nuisance separation 在系统中各司其职
- 通过 probe、retrieval、正式下游任务三条证据链共同证明模型价值

这份文档已经达到“可直接进入重构实现”的粒度。

下一步如果继续推进，实现工作应从：
- `types.py`
- `image_frontend.py`
- `spectrum_frontend.py`
- `shared_trunk.py`
开始。
