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

15 项单元测试、shell 语法及真实图接口检查通过。所有新部件图的旧字段逐项保持，原读取器可读；Hypersim 中两台功放 id=3、11 保持独立。之前原 ScanNet 分割一帧也实测通过，输出仍为 256/384 维。

图、逐帧结果、中文运行日志、机器可读摘要和测试输出全部位于外部 `partaware_v1`。模型环境使用隔离的 `.venv-vlpart`，避免改动原 conda 的 timm；具体安装见 `docs/PARTAWARE_SETUP.md`。

## 6. 结论与研究高度

交付已具备可运行、可消融、兼容旧接口的部件建图基线。未有人工真值，轨迹减少、确认数量和运行成功均不能证明精度提升。叠图曾出现桌架被标为桌腿、重复桌面；关联模式产生更多节点，也可能增加重复。因此默认算法仍保留 legacy。

下一阶段固定前端，在多个 ScanNet/Hypersim 场景按场景隔离评测实例/部件精确率与召回率、误合并/重复率、part_of 正确率和几何误差，分别消融颜色、SAM、去噪及包含清理。若要形成独立算法贡献，应聚焦局部观测归属与粒度不一致，研究多假设归属、可撤销合并及跨扫描对应驱动的结构修正；该联合算法尚未实现。只有可重复的精度收益才支持替换默认算法。

本轮新增代码、注释及 README 扩展使用英语，综合报告和运行消息使用中文。
