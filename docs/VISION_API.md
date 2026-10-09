# GPT 视觉接口、v12缓存与SAM3接入

## 接入状态

2026-10-08 已核实卡扣 AI 官网配置及接入教程的 API 域名为
`https://api.kakouai.com`，OpenAI 兼容客户端使用
`https://api.kakouai.com/v1`。使用用户本地密钥查询 `/models` 后，选择实际返回的
`gpt-6.1-sol`，Responses 流式图像请求已通过真实 Hypersim 图片测试。
这是中转站报告的模型 ID，本项目不能独立证明其上游实现。

Hugging Face 的 SAM3 权重访问此前被作者拒绝。本轮改用 ModelScope 公开
facebook/sam3 仓库的 sam3.pt，平台 SHA256 已核验，固定 Meta 官方源码本地推理。
来源元数据记录为 USER_UPLOAD，不把命名空间当作独立官方身份认证。
SAM3 在独立 WSL Conda 环境 sg-sam3 中运行，两个场景的真实单帧推理及
原DINO/SBERT接口已通过，完整场景精度仍待验收。它只处理可见二维掩码，
不能声称解决隐藏三维补全。

## 配置与直接命令

代码使用现有 WSL `scannet-sg` 环境的 OpenAI SDK，不安装或升级依赖。
本机配置位于项目根目录 `.gpt-vision.local.json`，密钥位于
`.local-secrets/kakou_api_key.txt`。两者均被 Git 忽略，密钥文件权限为 600。
不要将密钥放到命令参数、报告或原始响应中。通用无密钥模板见
`docs/vision_api.example.json`。

```bash
cd /home/lewisliu/PartAware-SG
/home/lewisliu/miniconda3/envs/scannet-sg/bin/python scannet/script/vision_api.py \
  --config .gpt-vision.local.json --list-models
```

单张图片测试不产生三维成品；测试的响应会被完整识图过程复用：

```bash
/home/lewisliu/miniconda3/envs/scannet-sg/bin/python scannet/script/vision_api.py \
  --config .gpt-vision.local.json \
  --manifest /home/lewisliu/datasets/scannet-sg-input/hypersim/ai_001_010_v3/manifest.json \
  --cache-root /home/lewisliu/datasets/scannet-sg-processed/gpt_vision_cache \
  --limit 1
```

本轮只构建v12，直接使用两场景已完成的GPT类别缓存。默认原始版仍用RAM，
默认v11仍为原有Qwen缓存；已经生成的GPT原始版/GPT v11只作为历史对照。
以下入口不会新增GPT或Qwen识图，缓存不完整就停止。输出根目录选一个新的
实验目录，以免覆盖此前的完整结果：

```bash
/home/lewisliu/miniconda3/envs/scannet-sg/bin/python scannet/script/run_gpt_comparison.py \
  --cache-root /home/lewisliu/datasets/scannet-sg-processed/gpt_vision_cache \
  --output-root /home/lewisliu/datasets/scannet-sg-processed/gpt_sam3_v12
```

也可重复指定 `--manifest` 选择场景，不改变输入采样。运行失败后，已完成
SAM3推理和成品按哈希校验复用；不重新调用VLM。完整模型与输入变化应使用
新的缓存目录，不能覆盖原结果。网络请求失败没有隐式重试；已收到但
不完整或无法解析的响应保留并停止，不能当成成功的空类别结果。

## 原接口与组件挂接

`vision_api.py` 读取原 `scannet_sg_input` manifest，不改变 RGB、深度、位姿
或帧编号。返回 `refined_instance/<frame_id>.json`，继续使用原
`objects: [{name, description}]` 合约。GPT 提供类别与外观描述，未提供可信
三维尺寸、隐藏结构或像素级实例掩码。墙、地面与天花板的表面类别排除；
其他平面物体不因 GPT 将其放进表面栏就自动丢弃，还须接受几何验收。
解析规则对两栏均检查名称的末尾名词，因此 `tiled wall` 归入房间边界，
而 `wall-mounted cabinet`、`floor lamp`、`ceiling light` 与 `countertop`
仍保留为物体。规则版本记录为 `envelope_noun_exclusion_v2`；只改变解析
时从相同原始响应重建类别，不重复付费识图。旧解析的缓存禁止直接绑定。

默认主流程增加 `RECOGNITION = gpt_recognition` 注册点。只有明确提供
`--recognition-config` 才获取或绑定 GPT 缓存；没有配置时继续使用旧缓存
或固定词表。删除该注册项及导入即可撤除 GPT 组件。原 ScanNet 文件夹
与 Hypersim manifest 接口保留，256维物体视觉、384维语义、1024维部件
特征互不混用。

| 默认版本 | 类别输入 | 后续构建 |
| --- | --- | --- |
| 原始版 | RAM | GroundingDINO/SAM、原 C++ 融合与原图后处理 |
| v11 | 既有Qwen缓存 | Florence/SAM、YOLOE-S、现有实例共识与实测修复、VLPart和正式几何发布 |
| v12 | 既有GPT缓存 | SAM3概念实例与局部放大、原DINO/SBERT特征、同一三维构建和发布接口；完整精度待验收 |

已有GPT原始版/GPT v11保留为历史补充对照，不作为默认识图来源。
比较配置在独立子进程中应用，不改写历史组件注册。v12分支集中由
`sam3_frontend.segment`挂接，代码注册改回`fovea`即可撤除SAM3。
完整ScanNet/Hypersim输入及公共图节点字段继续使用原接口。

## 缓存原理与审计

每场景保留 `vlm_responses`、`vlm_parsed`、`refined_instance` 和
`recognition_provenance.json`。识图请求指纹为：

$$
H_t=\operatorname{SHA256}(I_t),\qquad
Q=\operatorname{SHA256}(\text{model, URL, protocol, prompt, request settings}).
$$

只有图像哈希、请求指纹、数据集、场景与帧编号全部一致才复用原响应。
完整缓存绑定构建前再校验帧列表及原始响应哈希；单张测试不能被误用成
全场景缓存。场景级文件锁保证多个构建请求不会同时识别同一批帧，有限
线程只加速未缓存的不同帧。Responses 原始流事件也保留，避免第三方
扩展事件导致 SDK 高层解析失败后丢失可用结果。

三个构建版本从同一识图缓存获取语义，构建端新增 API 调用为零。
完成缓存的预期规模是 ai_001_002 的99帧与 ai_001_010 的100帧，共199份
有效响应；失败请求的可能计费由中转站决定，不能用有效响应数推断账单。
构建与类别请求均不读取 GT；GT 只用于完成后的独立评价。

## 评价与限制

所有版本沿用相同可观测 GT 范围、框生成及排序匹配协议，记录
AP25/AP50/AP75、MVO25/MVO50、一对一 TP/FP/FN、数量比与绝对对数误差。
这是项目适配的 Hypersim 观测几何框评价，不是官方 benchmark 分数。
结果位于各输出目录的 `evaluation.json`，总表位于
`scannet-sg-processed/gpt_comparison_summary.json`。

GPT 可以提供新的类别证据；能否纠正“盘子/餐巾”等错误须由实验验证。
它不自动解决一摞盘子被分为一个实例，
也不生成真实完整物体点云。当前双层细实例试验已完成两场景；第二场景
AP50 为11.82%，仍不足以称高精度。TripoSR、Hunyuan3D 和 MGPC 形状
试验均未证明通用补全可靠，严格验收下不能宣称薄片问题已解决。

## 2026-10-08：GPT 缓存恢复与构建适配

识图成功帧不重新请求。显式失败恢复命令可以加 `--retry-max-output-tokens 16384 --retry-transport-failures --new-max-output-tokens 16384`，只能恢复已保存的失败响应或首次请求，实际请求预算与失败 SHA256 记录在缓存内。`--recover-non-stream` 只适用于缺终止事件的失败流；本次非流恢复实际超时，没有把它当作成功结果。完整缓存不会因为恢复参数改变而重付费识图。

GPT v11/v12 对比配置在 `run_gpt_comparison.configure_profile` 注册 `OWNERSHIP_VALIDATION`，并传递给独立模型进程。`visibility_ownership.prune` 处理局部地板反证、唯一完整实测物体锚定与重复粗残片；`observed_consensus.refine_verified_surface` 只净化强实测核心附近的反复否定边缘。移除导入与注册即可拆卸，原始对照与普通默认注册不启用待两场景验收的新组件。数学、接口及最新结果见综合报告2.5、6.6及21章。

SAM3接入前第一场景已验收：历史GPT v11/v12均10个预测、10个GT，AP25/AP50为100%，AP75为80%，相较既有Qwen版62.5%提高17.5个百分点。第二场景GPT v12已完成，为98/109、AP25/AP50/AP75为22.33%/11.83%/2.94%；整体准入未通过，不能宣称两场景都提高。独立 `compare_evaluations.py` 检查相同GT口径和真实成品哈希，要求AP、数量误差、TP/FP/FN均不退步且至少一项改善，GT不会参与构建。

## 2026-10-09：Florence有界评分修复

第二场景一帧有164个描述候选，旧128个硬上限触发中止。默认Florence改为最多8个裁剪一批，对全部候选评分，继续原NMS和SAM；输入类别、检测特征与评分目标不截断。共享模型在同一帧各类别间驻留显存，阶段结束后再换出，不同模型仍顺序调度。真实4个裁剪与串行评分的平均对数似然最大差为0.003016，耗时2.229秒降至0.820秒；浮点结果不宣称逐位相同，最终仍须AP核验。

当前158项完整离线测试通过，两个场景的SAM3真实单帧模型推理、DINO256维、SBERT384维、标签像素一致性核验通过。新增识图请求为0。SAM3完整场景结果另存新的v12_sam3r1目录，未完成验收前不覆盖此前的已验证成品。
