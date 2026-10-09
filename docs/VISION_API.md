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
| v12 | 既有GPT缓存 | SAM3概念实例与局部放大、原DINO/SBERT特征、同一三维构建和发布接口；第一场景全面验收，第二场景保留AP与层级改善 |

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

历史GPT对比分支从同一识图缓存获取语义；当前仅运行v12，其构建端新增API调用为零。默认原始版/v11仍分别为RAM/既有Qwen。
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

当前178项完整离线测试通过，两个场景的SAM3真实单帧模型推理、DINO256维、SBERT384维、标签像素一致性核验通过。新增识图请求为0。SAM3完整场景连续修订另存独立目录，历史成品不覆盖。当前第一场景sam3r7、第二场景sam3r3已完成。

## 2026-10-09：WOC连续修订与第一场景验收

SAM3首次完整99帧的第一场景输出13个物体，AP25/AP50/AP75为100%/96.33%/85.50%，整体回退，未直接成为默认产物。随后持续从同一原始融合点、轨迹及二维观测重建：共享表面去重、跨名称图像身份及相机基线、细长平面评分、稳健主平面末端面关联。最终v12_sam3r5为10/10、100%/100%/100%，在三个框阈值下TP/FP/FN均为10/0/0。公共接口、19个确认部件及15条part_of关系核验通过，实际原版Open3D预览通过。第一场景因此保留SAM3及WOC；第二场景也已完整运行，当前AP25/AP50/AP75为32.21%/18.86%/6.17%；FP相对历史基线仍较多，不能套用第一场景全面验收结论。

WOC挂接在v12的WHOLE_OBJECT_VALIDATION，仅删除导入/注册即可拆卸。它复用既有1024维RN50辅助图像特征作身份证据，公共256维DINO/384维SBERT及部件特征不混用。所有归并均为实测表面，不生成隐藏点；GT仅进入独立评价。数学原理见报告2.6，接口见6.7，每次算法改变及完整结果见22章。
## 2026-10-09：原生掩码与BHA部件层级

原生SAM3整图重叠掩码只核验已有缓存签名和RGB哈希，不重新推理，不重新
调用识图API。`native_assembly.reconcile`在多帧完整支持和实测接触后关联
残片，v12的`SURFACE_ASSEMBLY`导入/注册是唯一拆卸点。

BHA的`part_body_assembly.construct`挂接在v12的`MEASURED_REFINEMENT`，
在已有VLPart部件之后执行。13块柜体薄面归到唯一完整实测体，同时保留
可查询面板节点和原点，正式`part_of`关系进入同一scene_graph。它没有把
几何部件伪装成VLPart视觉预测，也不保证每块面板就是完整的单柜。

第二场景139→126个独立物体，AP25/AP50/AP75由29.74%/17.52%/5.98%
提高至32.21%/18.86%/6.17%，局部严格准入通过；相对旧GPT/Qwen全指标
准入仍未通过。逐点核验面板与其他物体坐标保持。第一场景仍10/10、
100%/100%/100%。两个现有场景新增GPT/Qwen请求为0。
数学、层级评价口径与两场景可直接修改的可视化命令见综合报告2.7/2.8及22章。
