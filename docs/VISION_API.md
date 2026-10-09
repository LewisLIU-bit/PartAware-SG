# GPT 视觉接口、v12/v13共用缓存与SAM3接入

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
原DINO/SBERT接口已通过，两个完整场景已完成独立评价。它只处理可见二维掩码，
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

最新入口构建v13，直接使用两场景已完成的GPT类别缓存；v12保留历史参照。默认原始版仍用RAM，
默认v11仍为原有Qwen缓存；已完成GPT原始流程作为独立基线参与原始流程和v12/v13对比。
以下入口不会新增GPT或Qwen识图，缓存不完整就停止。输出根目录选一个新的
实验目录，以免覆盖此前的完整结果：

```bash
/home/lewisliu/miniconda3/envs/scannet-sg/bin/python scannet/script/run_gpt_comparison.py \
  --cache-root /home/lewisliu/datasets/scannet-sg-processed/gpt_vision_cache \
  --output-root /home/lewisliu/datasets/scannet-sg-processed/gpt_sam3_v13
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
| v12 | 既有GPT缓存 | SAM3概念实例与局部放大、原DINO/SBERT特征、同一三维构建和发布接口；第一场景全面验收，第二场景保留AP与层级改善 |
| v13 | 与v12同一GPT缓存 | 复用SAM3原生掩码及源观测，HMG整体保护、FDR/RSI精细实测与重复实例、PCB实测面定向、VPA视觉部件锚定 |

GPT原始版保留为独立对照，与v12复用同场景缓存。旧GPT v11中途成品已清理。
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

GPT原始与v12从同一识图缓存获取语义；既有两套成品参与对比，新增构建使用v13，其构建端新增API调用为零。默认原始版/v11仍分别为RAM/既有Qwen。
完成缓存的预期规模是 ai_001_002 的99帧与 ai_001_010 的100帧，共199份
有效响应；失败请求的可能计费由中转站决定，不能用有效响应数推断账单。
构建与类别请求均不读取 GT；GT 只用于完成后的独立评价。

## 评价与限制

所有版本沿用相同可观测 GT 范围、框生成及排序匹配协议，记录
AP25/AP50/AP75、MVO25/MVO50、一对一 TP/FP/FN、数量比与绝对对数误差。
这是项目适配的 Hypersim 观测几何框评价，不是官方 benchmark 分数。
结果位于各输出目录的 `evaluation.json`，总表位于
`scannet-sg-processed/final_results.json`，源码同步索引为 `docs/sam3_woc_results.json`。

GPT只提供类别证据，不能自动分离完全遮挡的堆叠盘子，也不生成隐藏三维表面。当前通用补全仍未通过可靠性验收，不能宣称薄片问题已解决。

独立 `compare_evaluations.py` 核对相同GT口径和真实图/点云哈希，要求AP、数量误差、TP/FP/FN均不退步且至少一项改善。原始GT只用于评价，不进入构建。报告第19章保留RAM原始、GPT原始、Qwen→DINO和最终GPT v12；第一场景RAM缺少同一99帧口径评价，不虚构可比分数。

## 当前最终v12与数据目录

两场景最终结果位于`scannet-sg-processed/partaware_v12/hypersim/`下。
ai_001_002为10/10、AP25/AP50/AP75 100%/100%/100%；ai_001_010为
126/109、32.21%/18.86%/6.17%。第二场景严格阈值FP仍偏多，全面准入未通过。
13块柜体面板保留为几何部件，其他点坐标不变。178项回归及正式接口通过。

GPT原始流程位于`gpt_original_v1/hypersim/`，参与RAM原始与最终v12比较。
Qwen→DINO对照位于`qwen_dino_v1`，RAM对照位于`ram_original_v1`。
每个历史版本只有`partaware_vN`一套最终包；中途SAM3/补全试验不保留。
已有GPT共享缓存和199帧SAM3缓存保留，构建不重新调用GPT/Qwen。

WOC、原生掩码接触关联分别挂接v12的WHOLE_OBJECT_VALIDATION、SURFACE_ASSEMBLY。
BHA挂接MEASURED_REFINEMENT，将唯一完整实测体边界上的源分实例保留为
可查询面板，并同步parts、part_of、轨迹、点云和正式空间关系。
删除对应导入/注册即可拆卸；它不能推断完整单柜或隐藏柜深。
数学原理见综合报告2.6–2.8，接口见6.7，最终结果见18章，比较见19章，命令见21章。

## 最新最终v13与直接复用

v13最终只有`partaware_v13/hypersim/ai_001_002`和`ai_001_010`两场景。
AP25/AP50/AP75分别100%/100%/100%与42.02%/25.97%/9.91%。
第二场景在三个阈值的TP/FP/FN均改善，但数量误差从0.1449变为0.2065，
不宣称全面门控通过或通用三维补全成功。原始GPT对照仍保留，并参加第19章比较。

现有正式结果及其签名已登记；`run_gpt_comparison.py`默认只选择v13，
已有正式文件会经图/点云哈希核验直接复用，不重新识图。
从v12原始融合重建的可修改单条命令见README。
新组件均在v13代码注册分支独立挂接；数学、接口、GT说明分别见综合报告
2.9、3.15–3.16、5.3、6.8–6.9，最终逐物体诊断见20章。
