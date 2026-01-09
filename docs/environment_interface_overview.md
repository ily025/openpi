# 环境接口与 Aloha 模拟示例详解

下面按调用顺序说明 `examples/aloha_sim/env.py` 如何通过 Gymnasium 包装机器人仿真，并把观测/动作转换成策略可消费的格式，同时补充接口基类的上下文。

## 通用环境协议

- 环境抽象基类位于 `packages/openpi-client/src/openpi_client/runtime/environment.py`，定义了推理/训练时环境需要实现的 4 个方法：`reset`、`is_episode_complete`、`get_observation`、`apply_action`。策略代码只依赖这组接口，不关心具体仿真或真实机器人实现。

## Aloha 模拟环境的实现要点

实现文件：`examples/aloha_sim/env.py`，类名 `AlohaSimEnvironment`。它继承上述基类并覆写所有抽象方法，承上（环境仿真）启下（策略推理）。

### 1) 初始化：绑定 Gymnasium 任务与随机种子

- 构造函数创建 Gymnasium 环境：`self._gym = gymnasium.make(task, obs_type=obs_type)`，`task`/`obs_type` 由外部传入，默认观测类型 `"pixels_agent_pos"`。同时为 NumPy 和内部随机生成器设定种子，保证复现性。

### 2) reset：拉起新一局并缓存第一帧观测

- 调用 `self._gym.reset(seed=int(self._rng.integers(2**32 - 1)))` 触发仿真环境重置，并从 Gymnasium 返回初始观测。
- 将 Gymnasium 的观测转换成策略期望的键格式：`self._last_obs = self._convert_observation(gym_obs)`。这一步把相机像素和机器人状态整理成统一字典，供策略读取。
- 标记当前 episode 未结束，累计奖励清零。

### 3) get_observation：提供最近一次观测

- 策略侧会周期性调用 `get_observation()`；实现直接返回在 reset/step 阶段缓存的 `self._last_obs`。如果还未 reset，会抛出异常提醒调用顺序错误。

### 4) apply_action：将策略输出的动作写入仿真

- 策略输出的动作格式为 `{"actions": <np.ndarray>}`，直接传给 `self._gym.step(action["actions"])`。
- 从 Gymnasium 的 step 返回值中读取下一帧观测、奖励和终止标志 (`terminated`/`truncated`)；调用 `_convert_observation` 再次做观测转化，并更新 `_done` 状态与累计奖励。

### 5) 观测转换细节 `_convert_observation`

- 视觉通道：从 Gymnasium 观测中取出 `gym_obs["pixels"]["top"]`，经过
  1) `image_tools.resize_with_pad(img, 224, 224)` 保持长宽比地缩放到 224×224，
  2) `image_tools.convert_to_uint8(...)` 统一像素类型，
  3) `np.transpose(img, (2, 0, 1))` 将通道顺序从 `[H, W, C]` 变为 `[C, H, W]` 以符合策略模型期望。
- 状态通道：直接暴露 `gym_obs["agent_pos"]`，以 `"state"` 键返回。
- 返回结构：
  ```python
  {
      "state": <agent_pos>,
      "images": {"cam_high": <3x224x224 uint8 image>},
  }
  ```
  这与策略推理代码中读取的观测键保持一致（相机键 `cam_high`、可选状态向量）。

## 与策略推理的衔接

- 策略入口 `Policy.infer` 会调用环境的 `get_observation`，并将返回的字典传入 `preprocess_observation(None, observation, train=False)` → Pi0 模型。Aloha 环境正是通过上述键/形状契约，确保 Gymnasium 的像素和状态能被直接送入模型，无需额外转换逻辑。
