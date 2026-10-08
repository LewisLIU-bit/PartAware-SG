# GPT 视觉接口、缓存与三版本对比

## 接入状态

2026-10-08 已核实卡扣 AI 官网配置及接入教程的 API 域名为
`https://api.kakouai.com`，OpenAI 兼容客户端使用
`https://api.kakouai.com/v1`。使用用户本地密钥查询 `/models` 后，选择实际返回的
`gpt-6.1-sol`，Responses 流式图像请求已通过真实 Hypersim 图片测试。
这是中转站报告的模型 ID，本项目不能独立证明其上游实现。

SAM3 权重访问被作者拒绝，用户已决定停止使用。当前细实例试验使用公开
YOLOE-v8-L 权重、切片与区域放大，不依赖 SAM3。已经运行的局部 SAM 研究
使用原有 SAM ViT-H；其自动碎片路线未通过验收，没有替代主流程。

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

两场景各用一份类别缓存，并生成原始、v11、v12 三种构建结果：

```bash
/home/lewisliu/miniconda3/envs/scannet-sg/bin/python scannet/script/run_gpt_comparison.py \
  --config .gpt-vision.local.json \
  --cache-root /home/lewisliu/datasets/scannet-sg-processed/gpt_vision_cache \
  --output-root /home/lewisliu/datasets/scannet-sg-processed \
  --recognition-workers 1
```

也可重复指定 `--manifest` 选择场景，不改变输入采样。运行失败后，已完成
识图和成品按哈希校验复用；不重新调用 Qwen。完整模型与输入变化应使用
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

| 对比版本 | 类别输入 | 后续构建 |
| --- | --- | --- |
| GPT 原始版 | 共用 GPT 缓存 | GroundingDINO/SAM、原 C++ 融合与原图后处理；不启用 PartAware 改进组件 |
| GPT v11 | 相同缓存 | Florence/SAM、YOLOE-S、现有实例共识与实测修复、VLPart 和正式几何发布 |
| GPT v12 | 相同缓存及 v11 粗观测 | 追加 YOLOE-L 整图/切片/放大细实例；粗细观测分层，继续同一三维构建与发布接口 |

这里的 GPT 原始版仅替换原 RAM 类别获取，不等于已经冻结的 RAM 原始
对照。比较配置在独立子进程中应用，不改写默认组件注册，也不使用
运行参数让默认算法相互混杂。v12 细分支集中由 `fovea.segment` 挂接，
切回 `FRONTEND = yoloe_frontend` 就可移除。默认经验证流程仍为 v11。

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
