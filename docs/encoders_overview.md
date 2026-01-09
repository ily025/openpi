# 视觉 / 语言编码器详解（文件与函数定位）

这份笔记按“入口配置 → 视觉编码 → 语言编码 → 在 Pi0 中的组装”展开，给出关键函数、主要参数以及源文件位置，便于对照代码阅读。

## 1. 编码器从哪里创建？
- **视觉 SigLIP**：`src/openpi/models/pi0.py` 的 `Pi0.__init__` 通过 `_siglip.Module(...)` 创建视觉编码器，显式指定 `variant="So400m/14"`、`pool_type="none"`、`scan=True`，并用 `img.lazy_init(...)` 初始化参数。【F:src/openpi/models/pi0.py†L66-L91】
- **语言/动作 Gemma (PaliGemma + Action Expert)**：同一构造函数内用 `_gemma.Module(configs=[paligemma_config, action_expert_config], embed_dtype=config.dtype, adarms=config.pi05)` 创建双专家 Transformer；`llm.lazy_init(..., method="init", use_adarms=[False, True] if config.pi05 else [False, False])` 决定是否为动作专家启用 adaRMS 条件分支。【F:src/openpi/models/pi0.py†L66-L100】
- **配置来源**：`paligemma_config`、`action_expert_config` 由 `openpi.models.gemma.get_config(...)` 返回，包含宽度、深度、LoRA 低秩设置等基础超参。【F:src/openpi/models/pi0.py†L66-L79】【F:src/openpi/models/gemma.py†L100-L110】

## 2. 视觉：SigLIP（`src/openpi/models/siglip.py`）
- **变体与默认超参**：`decode_variant("So400m/14")` 把字符串映射为 `width=1152`、`depth=27`、`mlp_dim=4304`、`num_heads=16` 等默认配置；如果传入 `variant="X/Y"` 形式会同时设置 `patch_size=(Y, Y)`。【F:src/openpi/models/siglip.py†L293-L360】
- **补丁提取与位置编码**：`_Module.__call__` 先将 NHWC 图像转为 `float32`，用 `nn.Conv(width, patch_size, strides=patch_size, padding="VALID")` 实现补丁化，再 reshape 为 `[B, HW, C]`。随后调用 `get_posemb(self, self.posemb, (h, w), c, "pos_embedding", jnp.float32)` 叠加可学习或 2D 正弦位置编码；`pool_type=="tok"` 时插入 CLS token。【F:src/openpi/models/siglip.py†L207-L235】
- **Transformer 编码栈**：`Encoder(depth=self.depth, ... scan=self.scan, remat_policy=self.remat_policy)` 构造多层自注意力 + MLP；`scan=True` 与 `remat_policy` 控制扫描/检查点节省显存，`dtype_mm` 允许半精度矩阵乘。【F:src/openpi/models/siglip.py†L238-L249】
- **输出形态与池化策略**：根据 `pool_type` 选择：`"gap"` 全局均值、`"map"` 多头注意力池化、`"tok"`/`"0"` 取首 token、`"none"` 保留所有补丁 token 并跳过池化。Pi0 使用 `pool_type="none"`，因此 `encoded` 的 `[B, HW, C]` token 将直接进入解码器。【F:src/openpi/models/siglip.py†L253-L272】

## 3. 语言与动作：Gemma / Paligemma 双专家（`src/openpi/models/gemma.py`）
- **词嵌入**：`Embedder.encode` 将 token ID 查表到 `input_embedding`，并乘以 `sqrt(embed_dim)` 做缩放；`Embedder.decode` 共享同一权重做输出投影（tied embeddings）。对应参数在 `Module.setup` 中以 `embed_dim=configs[0].width` 初始化。【F:src/openpi/models/gemma.py†L134-L155】【F:src/openpi/models/gemma.py†L343-L359】
- **注意力投影与 LoRA**：`Attention.__call__` 为每个专家构造 Q/K/V；当 `num_kv_heads < num_heads` 时使用分离的 `q_einsum` 与 `kv_einsum`，二者都接受 `config.lora_configs.get("attn")` 以附加低秩增量权重。位置编码通过 `_apply_rope(q|k, positions)` 注入旋转位置，`kv_cache` 参数允许在自回归推理中拼接历史键值。【F:src/openpi/models/gemma.py†L157-L215】
- **块结构与 adaRMS**：`Block.__call__` 先对每个专家执行 `RMSNorm(..., cond=adarms_cond[i])`，再跑注意力、残差、MLP（MLP 同样支持 `lora_config=config.lora_configs.get("ffn")`）。`adarms_cond` 来源于 Pi0 的时间嵌入，用于动作专家的条件归一化。【F:src/openpi/models/gemma.py†L283-L333】
- **Transformer 主体与 KV cache 形状**：`Module.__call__(embedded, positions, mask, adarms_cond, kv_cache=None)` 期望一个“每位专家一份”的嵌入列表，`mask` 被 reshape 为 `[B,1,T,S]` 后传入注意力；返回同样长度的编码列表与更新后的 `(k,v)` 缓存。`Module.embed` 则暴露仅嵌入步骤，供 Pi0 在组装前缀时调用。【F:src/openpi/models/gemma.py†L339-L412】

## 4. 在 Pi0 中如何把视觉/语言 token 拼在一起？
- **前缀构建**：`Pi0.embed_prefix` 迭代相机观测，调用 `self.PaliGemma.img(...)` 获得补丁 token，并复制 `obs.image_masks[...]` 形成 `input_mask`；若存在 `tokenized_prompt`，则用 `self.PaliGemma.llm(..., method="embed")` 得到语言 token 与掩码。所有 token 按“图像在前、语言在后”拼接，`ar_mask` 以 `False` 确保相互可见。【F:src/openpi/models/pi0.py†L105-L137】
- **动作后缀与 adaRMS 条件**：`Pi0.embed_suffix` 将归一化动作投到 `action_expert_config.width`，再与时间编码（`posemb_sincos`）混合；`pi05=True` 时走 `time_mlp_in/out` 生成 `adarms_cond` 传入 Gemma Block，关闭时则用双层 MLP 混合时间与动作。`ar_mask` 在动作部分首位设 `True`，禁止图像/语言 attends 到动作 token。【F:src/openpi/models/pi0.py†L139-L187】
- **注意力掩码生成**：`make_attn_mask(input_mask, mask_ar)` 根据 `ar_mask`（是否阻止当前 token 看到前序块）和 `input_mask`（padding 与否）生成 `[B,S,S]` 布尔矩阵，供 Gemma 模型使用，实现“图像/语言互看、动作只看自身与前缀”的模式。【F:src/openpi/models/pi0.py†L19-L45】

以上步骤使得 SigLIP 的视觉补丁 token 与 Gemma 语言 token 在 Pi0 的 Transformer 中组成统一上下文；动作专家在相同堆栈内共享注意力与缓存，但通过独立权重与 LoRA 适配器学习动作分布。

## 5. 语言 → 动作的承上启下（训练与推理全链路）

- **训练阶段：前缀 + 后缀一次前向**：`Pi0.compute_loss` 先拼好前缀（视觉+语言）与后缀（动作+时间）token，再把掩码联结后一次喂入 `self.PaliGemma.llm`，返回 `[prefix_out, suffix_out]`。动作流速 `v_t` 由 `action_out_proj` 线性投影得到并与目标速度场 `u_t` 做均方误差，实现 flow-matching 监督。这里的 `adarms_cond` 直接随后缀输入传入，以便 Block 的 `RMSNorm(..., cond=...)` 在动作专家上做条件归一化。【F:src/openpi/models/pi0.py†L188-L214】【F:src/openpi/models/gemma.py†L283-L333】
- **推理阶段：前缀缓存 + 后缀迭代**：`Pi0.sample_actions` 首次仅用前缀调用 `self.PaliGemma.llm([prefix_tokens, None], ...)` 获得 KV cache；随后在 `step` 闭包里，每个时间步都通过 `embed_suffix` 重算当前动作 token 与 `adarms_cond`，构造 `full_attn_mask` 让动作 token 能看到全部前缀和自身历史，再用缓存复用视觉/语言键值以节省算力。得到的 `suffix_out` 经过 `action_out_proj` 给出速度场，按欧拉积分更新 `x_t`，形成动作轨迹。【F:src/openpi/models/pi0.py†L217-L279】
- **语言提示如何影响动作**：语言 token 在前缀阶段通过 `self.PaliGemma.llm(..., method="embed")` 生成并存入 KV cache；`full_attn_mask` 使得后缀动作 token 查询该 cache，因此动作专家可直接 attend 到语言键值。mask 的首位 `True` 则阻断动作 token 彼此的前瞻，保持自回归流匹配设置。【F:src/openpi/models/pi0.py†L127-L186】【F:src/openpi/models/pi0.py†L239-L268】
- **策略入口如何调度这一流程**：`policies/policy.py:Policy.infer` 接收环境观测后做批处理和变换，实例化 `_model.Observation`，再调用 `model.sample_actions(...)`。因此外部只需提供状态、相机、提示等观测，语言到动作的衔接逻辑全部封装在 `Pi0` 内部的前缀/后缀嵌入与解码循环中。【F:src/openpi/policies/policy.py†L24-L107】
- **`_model.preprocess_observation(None, observation, train=False)` 里的 `_model` 是什么？** `pi0.py` 顶部将 `openpi.models.model` 模块别名为 `_model`，推理时调用其中的 `preprocess_observation` 进行尺寸校正与缺省 mask 填充；第一个参数传入 `None` 表示不需要随机增强（训练态才会用 RNG 做增广）。这一步把 `Policy.infer` 准备好的 Observation 变为 Pi0 期望的格式，随后才进入前缀/后缀嵌入与解码循环。【F:src/openpi/models/pi0.py†L10-L18】【F:src/openpi/models/pi0.py†L217-L225】【F:src/openpi/models/model.py†L144-L203】【F:src/openpi/policies/policy.py†L71-L100】
