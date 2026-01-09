# 数据加载流程速览（面向新手）

下面用“流程图 + 关键入口”方式，解释 `src/openpi/training/data_loader.py` 如何把原始数据整理成模型能训练/推理的批次。

## 1. 配置驱动
- 入口函数：`create_data_loader(config, …)`。
- 配置来源：`TrainConfig`（例如 `scripts/train.py` 传入），其中 `config.data` 决定数据源、归一化与变换，`config.model` 提供动作长度（`action_horizon`）和输入规格。
- 数据源分支：
  - **本地/LeRobot**（`data_config.rlds_data_dir` 为空）→ `create_torch_data_loader`。
  - **RLDS/DROID**（`rlds_data_dir` 非空）→ `create_rlds_data_loader`。

## 2. 构建原始 Dataset
- LeRobot / 本地：`create_torch_dataset`
  - 当 `repo_id="fake"` 时返回 `FakeDataset`，用于快速单元测试。
  - 否则使用 `lerobot.common.datasets.LeRobotDataset` 读取轨迹，并根据 `action_horizon` 展平/切片动作序列。
  - 如果开启 `prompt_from_task`，会追加 `PromptFromLeRobotTask` 变换，把任务名拼成文本提示。
- RLDS / DROID：`create_rlds_dataset`
  - 返回 `DroidRldsDataset`，内部已做随机采样与批处理，支持 `filter_dict_path` 过滤任务。

## 3. 串联数据变换
- 函数：`transform_dataset`（随机访问）/`transform_iterable_dataset`（可迭代）。
- 变换顺序：
  1. **repack_transforms**：把原始字段重命名/重排成模型预期结构。
  2. **data_transforms**：如裁剪、色彩抖动等数据增强。
  3. **Normalize**：用 `data_config.norm_stats` 做均值/方差或分位数归一化（可通过 `skip_norm_stats` 跳过）。
  4. **model_transforms**：模型专属处理，例如把动作序列分片。
- 变换通过 `TransformedDataset` 或 `IterableTransformedDataset` 包装；后者支持对已批处理的 RLDS 数据逐样本应用变换，再重新拼批次。

## 4. 打包成 DataLoader
- Torch 数据：`TorchDataLoader`
  - 处理 PyTorch 或 JAX 框架，支持 `torch.distributed.DistributedSampler` 划分全局批次。
  - 本地批次大小：
    - PyTorch：`batch_size // world_size`（若 DDP 已初始化）。
    - JAX：`batch_size // jax.process_count()` 并可应用 `NamedSharding` 做数据并行。
  - 迭代行为：支持 `num_batches` 限制；用 `_collate_fn` 把列表转为批量 numpy，再按框架转换为张量或 ShardedArray。
- RLDS 数据：`RLDSDataLoader`
  - 直接迭代 `DroidRldsDataset` 已经形成的批次，同样支持 `num_batches`，并在 JAX 下默认数据并行分片。

## 5. 统一返回类型
- `DataLoaderImpl` 是薄包装，只暴露 `__iter__` 和 `data_config()`，方便训练循环保存/恢复时记录数据设置。

## 6. 典型调用示例
```python
# scripts/train.py 片段
loader = create_data_loader(config, shuffle=True)
for batch in loader:
    observations, actions = batch  # 已按模型规格、归一化和变换整理
    ...  # 送入模型
```

这样，即使不熟悉 JAX/PyTorch，也能从配置 → Dataset → 变换 → DataLoader 的顺序理解数据是如何被整理并喂给 VLA 模型的。
