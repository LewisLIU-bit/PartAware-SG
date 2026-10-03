# PartAware-SG 综合研发与实验报告

更新：2026-10-03。本文合并原实验记录和本轮集成记录，保留与当前 ScanNet/Hypersim 主线有关的结论。完整旧记录及旧运行产物已移到 `datasets/scannet-sg-processed/partaware_v1/history`。

## 1. 目标、备份与目录

目标是提高实例与部件建图的可靠性，区分独立物体、真实部件、同一物体的局部观测；尤其避免把两台功放并入桌子。新增部件层与旧物体图共存，基础 ScanNet 建图和 Hypersim 清单接口保留，默认物体关联仍为 legacy。

- 核心项目：`/home/lewisliu/PartAware-SG`。
- 改造前全量备份：`/home/lewisliu/ScanNet-SG-backup-20261003-175312.tar`，包含隐藏文件、Git 状态、旧修改、权重和构建产物；2217 个条目逐项校验。SHA-256：`b183483537ace34fa854679fb6d4eb7d206312bae6d955d4bf3fd44bc970e8f6`。
- 原库已恢复到 GitHub 提交 `52766bba5b4bf7e4773480d1b6c9703c8d8069da`，受版本管理代码保持原状。
- 当前仅保留 ScanNet/Hypersim 专属路线；移除 3RScan、KITTI-360 适配器、词表和专项测试文档。其他原始数据目录未改动。
- 第一版结果统一存放在 `/home/lewisliu/datasets/scannet-sg-processed/partaware_v1`，与其他实验同级。项目中不保留本轮运行结果、旧编译对照、代码备份及临时下载文件；必需的模型、依赖和兼容补丁保留。

已阅读保存 HTML 中的全部可见对话；页面未加载的更早历史不在文件内。归档位于上述外部 history/records。

## 2. 阅读与实际接入

| 来源 | 接入或核对内容 |
| --- | --- |
| [OpenSGA](https://arxiv.org/abs/2605.10484) / [ScanNet-SG](https://github.com/tud-amr/ScanNet-SG) | 核对标签、DINO 256 维特征、SBERT 384 维特征、投影、物体图格式 |
| [OP3DSG](https://arxiv.org/abs/2606.29786) / [代码](https://github.com/AutoCompSysLab/OP3DSG) | 内置官方物体—部件知识库，改编几何/语义/颜色关联和物体—部件组织 |
| [VLPart](https://arxiv.org/abs/2305.11173) / [代码](https://github.com/facebookresearch/VLPart) | 官方 Swin-base LVIS/PACO 模型、动态类别分类器、原生掩码 |
| [ConceptGraphs](https://arxiv.org/abs/2309.16650) / [代码](https://github.com/concept-graphs/concept-graphs) | 掩码区域特征、方向性几何覆盖，可选 DBSCAN 和包含掩码清理 |

代码版本和模型校验见 `docs/sources.lock.json`、`docs/model_checksums.json`。这里只改编 OP3DSG 的先验建图部分，未接入其 LLM 统一推理，不宣称完整复现。未找到可核实的 g-ch/OpenSGA 公共匹配推理仓库，未创建猜测接口。

旧阶段已有 Qwen/联合打分、Hypersim 输入、地板过滤及重复实例处理；本轮保留这些代码。此前 0.395534 候选与 0.39 阈值实验只属于旧结果，本轮不据此改动默认前端阈值。

## 3. 接口与改动

入口为 `scannet/script/run_partaware.py`：`--image-dir` 读取 ScanNet，`--manifest` 读取包括 Hypersim 的既有 `scannet_sg_input` 清单；`--processed-scene` 提供旧物体图及真实实例映射，`--output` 指向新的外部结果目录。

输出 `partaware_graph.json` 保留所有旧图字段，附加 `part_nodes`、`part_relations` 和来源信息。逐帧布尔掩码、观测点云、参数与中文运行日志分别保存为 NPZ、NPY、`run_config.json`、`run_zh.jsonl`。部件使用独立的 CLIP RN50 1024 维特征，不写进物体的 256/384 维特征；旧 TopologyMap 可正常读取物体字段。

主要保护：父节点必须有类别及掩码覆盖证据；歧义保持未绑定；不同父物体不融合；同帧可见观测不能互相合并；默认至少两帧确认后才形成 part_of。功放保持独立实例。

可选项：SAM 框精修、`--dbscan-eps` 最大簇去噪、`--subtract-contained` 同真实父实例下的包含清理、`--erode-pixels` 边界收缩。附加清理默认关闭。C++ 新增 `--association_mode legacy|op3dsg`，原位置参数及 manifest/filter_floor 接口保留。旧 SentenceTransformer 构造参数兼容修复不更换模型或维数。

## 4. 数学原理与边界

**RGB-D 投影。** 深度为光轴 Z，$z=d/s$，$s$ 为深度缩放因子。利用深度内参：

$$p_c=[(u-c_x)z/f_x,\ (v-c_y)z/f_y,\ z]^T,\qquad p_w=Rp_c+t.$$

彩色/深度使用各自内参对齐掩码采样，不用简单图像缩放代替投影；沿用原库已配准 RGB-D 假设，坐标为米，位姿为相机到世界。无深度和无观测区域不补造几何。

**局部观测关联。** 令 $A$ 为当前点云、$B$ 为轨迹，半径 $r$ 内的方向覆盖率为：

$$c(A\to B)=\frac{1}{|A|}\sum_{a\in A}\mathbf1[\min_{b\in B}\|a-b\|\le r],\quad G=\max(c(A\to B),c(B\to A)).$$

最大方向覆盖能让小范围观测匹配到大物体；也可能让相似局部区域产生误匹配，因此必须同时检查语义、父物体及同帧约束。部件默认 $r=0.03$ 米，$G\ge0.2$。

**语义与颜色评分。** 归一化区域特征的余弦为 $q=f_A^Tf_B$，转换为 $S=(q+1)/2$。颜色采用 LAB a/b 与两个 opponent 通道的归一化直方图，平均一阶 Wasserstein 距离 $D$ 转成 $C=1/(1+3D)$。当前部件评分为：

$$Q=2[(1-w)(G+S)+wC],\qquad w=0.6,\quad Q\ge1.5,\quad q\ge0.7.$$

这是适配实现，不等同于论文完整 Color Names 组合。颜色只提供辅助证据，不能绕过几何/语义门限。颜色轨迹更新为 $h\leftarrow0.7h+0.3h_{obs}$；视觉特征作归一化累计平均，点云合并后体素化。

**归属与确认。** 父掩码对部件的覆盖至少 0.2，前两名覆盖差默认至少 0.1，否则父编号为空。同帧排斥利用已观测帧集合，多帧确认要求不同帧数达到 2；这提供时序证据，但重复检测或持续误检仍可能被确认，不能把 confirmed 当成真值。

**清理。** 包含比例达到 0.9 且面积比超过 1.5 时，只有同真实父实例、不同部件标签才允许小掩码从大掩码扣除。DBSCAN 保留最大簇可清理背景，但可能删掉真实不连通部件，因此两项均作为独立消融。

## 5. 数据集实测

原始输入固定来自 `/home/lewisliu/datasets`。Hypersim 使用用户指定的 `scannet-sg-input/hypersim/ai_001_002`，100 帧中 98 帧有完整旧实例映射；`cam_02_0024`、`cam_03_0034` 缺少旧 updated_instance 文件，明确记录并跳过。ScanNet 使用真实 `scene0000_00` 的 30 帧。

| 分支 | 帧数 | 部件轨迹 | 多帧确认 | part_of | 耗时（秒） |
| --- | ---: | ---: | ---: | ---: | ---: |
| hypersim：VLPart 原掩码 | 98 | 30 | 20 | 17 | 20.09 |
| hypersim：SAM+去噪+包含清理 | 98 | 27 | 18 | 15 | 109.83 |
| scannet：VLPart 原掩码 | 30 | 14 | 9 | 9 | 13.85 |

C++ 扩大样本回归：Hypersim 旧二进制/新 legacy 均为 11 个物体节点，op3dsg 为 14 个；ScanNet 三者均为 4 个。两种数据的新 legacy 完整 JSON 都与改造前保留的旧二进制精确一致。这里的重新融合图与用于部件实验的旧清理后图属于不同阶段，节点数不直接混比。

v1 的 15 项单元测试、shell 语法及真实图接口检查通过。所有新部件图的旧字段逐项保持，原读取器可读；Hypersim 中两台功放 id=3、11 保持独立。之前原 ScanNet 分割一帧也实测通过，输出仍为 256/384 维。

图、逐帧结果、中文运行日志、机器可读摘要和测试输出全部位于外部 `partaware_v1`。模型环境使用隔离的 `.venv-vlpart`，避免改动原 conda 的 timm；具体安装见 `docs/PARTAWARE_SETUP.md`。

## 6. 结论与研究高度

交付已具备可运行、可消融、兼容旧接口的部件建图基线。未有人工真值，轨迹减少、确认数量和运行成功均不能证明精度提升。叠图曾出现桌架被标为桌腿、重复桌面；关联模式产生更多节点，也可能增加重复。因此默认算法仍保留 legacy。

下一阶段固定前端，在多个 ScanNet/Hypersim 场景按场景隔离评测实例/部件精确率与召回率、误合并/重复率、part_of 正确率和几何误差，分别消融颜色、SAM、去噪及包含清理。若要形成独立算法贡献，应聚焦局部观测归属与粒度不一致，研究多假设归属、可撤销合并及跨扫描对应驱动的结构修正；该联合算法尚未实现。只有可重复的精度收益才支持替换默认算法。

本轮新增代码、注释及 README 扩展使用英语，综合报告和运行消息使用中文。


## 7. 共用原版可视化（v2 修正）

v1 曾使用独立的简化部件绘图，没有完整复用原版的颜色、包围盒和交互；该实现已经替换。现在正常图和精修图均调用 `script/utils/result_visualization.py::visualize_map_with_nodes`，只通过 `build_part_overlay` 追加子节点和虚线父子边。

不开部件时，真实 Hypersim 数据的 41 个基础几何对象（含点云/颜色/节点/包围盒/关系）与提交 adaa808 中原绘图函数的数组及哈希逐项一致；开部件时这 41 个对象仍逐项一致，只追加 30 个几何：15 个已确认且有父实例的子节点与 15 条父子边。小节点采用父实例配色，物体关系保持原色，默认隐藏部件点云与无归属轨迹。诊断开关仍能显示全部 27 条部件轨迹及其点云。

两个入口均复用原 GUI/legacy 窗口，GUI 拾取包含物体与子节点。原版 Open3D 实际生成了截图并作图像检查；拾取回调已接入，未模拟真实鼠标点击。新增可选尾参数和 CLI 见接口文档，旧参数保留；`visualize_partaware.py` 是兼容包装器。20 项部件与显示回归检查通过；旧 Florence 参考配置及显式 DINO 回退也通过真实图像测试。

所有可视化命令均带用户要求的 `env -u WAYLAND_DISPLAY -u WAYLAND_SOCKET` 和 `XDG_SESSION_TYPE=x11 LIBGL_ALWAYS_SOFTWARE=true`。使用 tracking-ID 编码的 `instance_cloud*.ply`，避免把已经着色的 RGB 导出重新当实例编号解释。

## 8. Florence 默认入口与真实 v2 测试

基本物体流程默认采用 DINO 候选框 → Florence 裁剪区域描述证据 → SAM → 原 C++ 融合/JSON。ScanNet 文件夹入口及 Hypersim manifest 不变，256 维视觉特征与 384 维文字特征不变。`--grounding_backend dino` 或 runner 的 `GROUNDING_BACKEND=dino` 可复现原前端，显式 `--joint_config` 保留旧场景参考归一化与硬负样本逻辑。部件检测仍使用 VLPart/CLIP，并未声称已经由 Florence 替代。

便携默认不依赖本场景参考图；全部输入类别获得描述证据。只实例化一个长期 Florence worker，在已有 `sg-florence` 环境中运行（transformers 4.49.0 / timm 1.0.15 / einops 0.8.1），避免升级基础 RAM 或 VLPart 依赖。DINO、Florence、SAM 分时进入 GPU。修复了 transformers 把 Florence 条件导入 FlashAttention 当成必需依赖的问题，仅在本模型加载期间使用 eager attention 兼容处理；显式加载可信本地配置消除交互提示。缺失模型或推理错误会明确失败，不悄悄切换后端。

对候选裁剪区域 $I$，描述 $d=(y_1,\ldots,y_N)$ 的证据为普通 token 的平均条件对数似然：

$$\ell(d\mid I)=\frac{1}{N}\sum_k\log p(y_k\mid y_{<k},I,\texttt{<CAPTION>}),\quad m=\ell(d_c\mid I)-\max_{j\ne c}\ell(d_j\mid I).$$

描述只作为 decoder 目标，encoder 不看到目标答案。默认融合为 $s=\sigma(0.8\operatorname{logit}(p_{DINO})+0.2m)$。这只是可解释的未校准证据，不能作为准确概率。默认比较背景及少量易混类别，采用软证据；旧显式参考配置仍按其硬负样本规则运行。

真实测试先发现：短模板配合硬语义否决会误拒绝可见功放。该失败消融已保存在 v2/history；随后改为描述结构特征及软证据。真实 Hypersim 5 帧与 ScanNet 2 帧默认推理完成，并通过原 C++ 融合、generate_json、SAM 部件精修完成全链路。新物体图分别为 9 / 3 个节点，部件轨迹为 13 / 1 条；其旧图字段逐项一致。小样本及抽帧跨度会改变观测支持，不能与 v1 的 98 / 30 帧节点数直接比较。

v2 输出为 `datasets/scannet-sg-processed/partaware_v2/{hypersim/ai_001_002,scannet/scene0000_00}`，与其他实验同级。这是 7 帧端到端功能验证，未重跑完整 100 帧；v1 全帧结果和原始失败记录保留。新代码修改不会自动更新旧缓存。

功放诊断：v1 id=3、11 的中心相距约 0.58 米，原始画面有两台物理功放，且两帧 updated_instance 记录同时含这两个编号。左侧首帧仅有一个功放实例掩码，但边缘未完整覆盖，融合点云还含离散点。因此不能把“两个 amplifier 类别节点”直接判为重复实例；用户看到的左侧碎片需结合具体视角检查掩码缺失、类别竞争与跨帧投影误差。诊断数据和回投影保存在 v2，未强制合并两个节点，也未宣称已经消除所有碎片。

## 9. 论文启发与下一步建议（尚未完整实现）

结合当前 `PartFusion.add` 的同标签/同父实例关联、`select_parent` 的单帧覆盖率，以及 C++ 几何/语义融合，优先级如下。以下是迁移建议，不能当作论文算法已复现或新精度结果。

1. **先解决碎片与身份，再扩展语义。** [ConceptGraphs](https://concept-graphs.github.io/assets/pdf/2023-ConceptGraphs.pdf) 将几何覆盖和视觉相似度结合，并用多视角描述总结物体。建议每个物体保存少量高可见度裁剪，用 Florence 对多视角证据作稳健汇总；融合候选同时检查重投影掩码 IoU、深度一致性与几何覆盖。只有证据支持同一物理实例才合并，保留原编号映射、可撤销记录和独立完整实例的同帧冲突约束。Florence 的同类描述不能证明身份相同。
2. **不要让第一次归属锁死部件。** [Hierarchical and Holistic Open-Vocabulary Functional 3D Scene Graphs](https://arxiv.org/html/2605.15753v1) 将图像证据、多线索关联和时序归属优化结合。建议把 `select_parent` 的一次性父编号扩展为候选分布，累积多帧支持并保持歧义：$L_j=\sum_t w_t\,\operatorname{logit}(s_{tj})$，再结合熵正则与时间平滑选择父节点。这里只是拟议公式；当前 Florence 分数未经校准，不可直接代入概率 logit，必须先校准或使用有限幅度的 margin。避免相邻重复帧造成虚假高置信度。
3. **改进遮挡与区域污染处理。** [Open3DSG](https://arxiv.org/html/2402.12259v2) 利用深度检查可见性并聚合较好视角的特征。建议过滤遮挡/边缘裁剪，再评价功放与桌子的竞争；不能用透明桌面的大框覆盖作为语义归属依据。关系应由具体对象对及图像证据支持，不能把类别相似度直接当作 part_of。
4. **对代表特征做稳健聚合。** [HOV-SG](https://arxiv.org/html/2403.17846v2) 结合区域/掩码/全图特征，利用多数特征簇减弱噪声，并支持重叠片段图合并。建议保留原特征维数，将部件的单纯均值改为去异常视角后聚合；规范 table/desk 等同义标签与部件名，同时记录原标签。按父实例及几何门控去重，防止同帧真实多条桌腿被错误合并。

建议的消融顺序：原 DINO → 默认 Florence → 多视角 Florence → 投影一致关联 → 软父归属。固定帧清单、旧接口和人工真值，以实例 split/merge 率、ID 一致性、mask IoU、part_of 准确率、召回率及耗时评价。目标是减少错误碎片与归属且不牺牲真实物体召回；在这些指标通过前，不声称达到 SOTA 或以“节点更少”证明更精确。

最终源码哈希、原版几何一致性检查、模型加载兼容与原路线回归均记录在 v2。环境版本与本地模型源码哈希见 `docs/environment_florence.json`。
