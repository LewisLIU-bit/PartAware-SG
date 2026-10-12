# EviScene 高显存训练与自主诊断

本目录是研究原型，默认最终构建仍为冻结 v19。真实数据训练尚未启动。
2026-10-11：已完成四场景独立诊断、三种候选融合对照、验证网络合成
前向/反向、冻结主干检查和 MMEngine 配置生成。没有发布新的改进版本。
根目录 AGENTS.md 是代码结构、环境、问题和后续进度的维护入口。

## 最少训练路线与可选依赖

1. A / 可选ESAM局部适配：默认复用官方权重并冻结大骨干；有证据时才训练
   memory、pool、query decoder 和 merge head；再对选定主干层解冻。
   沿用类别无关实例头，200类语义只作原配置辅助监督，不能用来
   丢弃开放词汇候选。保留未微调 ESAM 和 v19 两条基线。
2. B / MATCH-3D 验证：实际 ESAM 查询、文本、局部几何、预测图上下文、
   独立 RGB-D 证据绑定为训练记录。先学习身份、语义匹配、存在性和掩码质量。
   同名不同物体在文本匹配上可同时为正，在身份关联上必须分开。
   名称不相容不等于物体不存在，遮挡未知不能作为幽灵负样本。
3. C / 条件补全：用有完整物体几何的独立数据训练文本/图条件解码器。
   先保持实测表面，再检验自由空间、邻居碰撞和跨视角一致性。
   ScanNet 重建只有观测表面，不能充当隐藏背面的完整真值。
   AdaPoinTr 已有几何先验可作初始化，条件化与重训尚未实施。

执行优先级为零新增训练算法循环 → 可选小型B验证器 → 可选局部A → C。
完整ESAM重训不作为必经步骤，每项保留冻结成品并单独消融。
高显存优先用于多帧与点级特征，不先扩张 VLM 或堆叠额外模型。
ESAM论文附录用额外开放词汇分类扩展CA掩码，本地权重不等于该完整模型。
因此分别验收实例AP、命名/语言定位、
细物体保持以及未见类别；类无关 AP 不能证明 open-set 语义准确率。

## 数据划分和规模

本地独立有标注 ScanNet 训练扫描只有 scene0000_00。官方 train 列表
1201项中缺1200项；单场景仅用于流程检查，不用于泛化结论。
先按官方划分补齐一个事先固定的多场景小批次做数据/梯度检查，
小型验证器先测固定多场景批次的数据效率，完整train微调只作为可选分支。
分别报告新增参数、独立扫描数、训练时长与显存。

scene0011_00、scene0015_00、scene0019_00、ai_003_007 和 ai_001_010
均是已查看诊断的开发场景，不能混入训练或冒称新模型未见测试。
同建筑/扫描变体按组隔离，不能将同一房间不同帧随机分到 train/test。
最终检验使用训练前固定、未用于算法选择的新场景；报告样本数和划分。
训练记录由训练集标注构造，推理图与图边不直接读 GT。
GPT缓存可复用；本轮准备与原型对照没有新增识图请求。

## 大机子到位后的顺序

1. 保存 GPU 型号/卡数、显存、驱动、CUDA、CPU/RAM、存储空间、
   源码和权重哈希。独立建训练环境，保留生产环境。
2. 检查 ESAM 的真实稀疏 CUDA 前向和反向。当前本地兼容环境是 CPU
   MinkowskiEngine；CPU推理通过不能证明GPU训练通过。
   官方旧环境使用 CUDA11.6/PyTorch1.13.1，不直接照搬到新架构GPU。
3. 转换官方数据为 scannet200-mv 所需点、超点、实例/语义和信息文件，
   对照 ESAM 的 docs/dataset_preparation.md，检查米制、位姿、标签与实例一一对应。
4. 用未改权重复现原生指标，再用短训练检查损失、梯度、峰值显存、
   checkpoint恢复。合成接口检查不替代这些真实数据检查。
5. 先单卡再 DDP；AMP 只在稀疏算子前向/反向与数值对照通过后打开。
   梯度累积的有效场景批量 = 每卡batch × 累积次数 × 卡数。

24GB候选：每卡1场景×2帧、累积4；48GB：1×8、累积4；
80GB：2×8、累积2。均是开始试验的配置，尚无实测显存保证。
正式学习率/有效批量按真实单卡结果和独立验证确定，不根据大机子
显存大小直接宣称能跑多少场景。

## 已有工具

- preflight_training.py：划分/数据/GPU与官方入口预检，不启动训练。
- generate_stage_A.py：从官方类别无关配置生成实际 MMEngine 配置，
  保留 oneformer3d 注册与原 hook；添加 EviSceneFreezeHook 和梯度累积。
- freeze_adapter.py：冻结参数，并保持 norm 统计处于 eval。
- evi_match.py：725890参数的验证头原型；输入维数256/384/32/64/12。
  文本与结构分别投影，共用泛称允许多正匹配；身份分支区分实际实例。
  当前尚缺真实查询导出、几何/图编码器、记录构造、训练器和校准器。
- mask_proposals.py：不读取 GT 的独立候选适配，不代表完整建图融合。
- audit_instances.py：独立 GT 诊断；与候选构建代码分开。
- test_*.py：关键行为/梯度检查，不是数据集训练或模型效果。
- training_plan.json：机器可读路线与开放词汇约定。

ESAM查询不能被同维DINO向量替代。图上下文不含目标自身标签/描述；
邻居语义亦须可追溯，遮蔽消融检查标签捷径。
存在性、文本相容度、IoU质量和补全质量分别学习/校准。

## 实际命令入口

在实验室 Linux 中定义 EVI_ROOT、ESAM_ROOT、ESAM_CKPT、TRAIN_PY、
TRAIN_DATA 和 SCANNET_META 为真实路径。TRAIN_DATA 应指已转换的
scannet200-mv；SCANNET_META 指包含官方训练/验证列表的Benchmark目录。
这些是迁移模板，当前未在实验室执行。

    "$TRAIN_PY" "$EVI_ROOT/tools/research/preflight_training.py" \
      --dataset-root /path/to/raw/scannet/scans \
      --train-list "$SCANNET_META/scannetv2_train.txt" \
      --val-list "$SCANNET_META/scannetv2_val.txt" \
      --esam-root "$ESAM_ROOT" --output /path/to/run/preflight.json

    "$TRAIN_PY" "$EVI_ROOT/tools/research/generate_stage_A.py" \
      --esam-root "$ESAM_ROOT" --checkpoint "$ESAM_CKPT" \
      --data-root "$TRAIN_DATA" --profile 48GB --pilot-epochs 2 \
      --output /path/to/run/stage_A_pilot.py

    cd "$ESAM_ROOT"
    PYTHONPATH="$ESAM_ROOT:$EVI_ROOT/tools/research:$PYTHONPATH" \
      "$TRAIN_PY" tools/train.py /path/to/run/stage_A_pilot.py

去掉 pilot-epochs 生成正式候选配置，但必须先通过实际数据检查。
没有自动下载完整数据/启动长任务或在本机8GB环境强行训练。

## 自主诊断与准入

四场景逐GT诊断和源深度可见性已保存；这只提供问题线索，不进入构建。
直接追加/并集/交集 ESAM 候选都使三个扫描合并 AP50/平均AP 退步，
因此未改变 v19。不要仅因 AP25 增加就把候选升级为正式算法。
后续优先：同类椅子误合并、部分结构可见范围、掩码质量排序、
源点与评估域不一致。目标域空掩码不自动等同幽灵。

按同场景、同输入和同评价比较 v19、ESAM、ESAM+A、再+B、最后+C。
加测小物体召回/实例保持、ID切换、误删/漏删、语义检索和校准；
有完整GT后才报告补全CD/F-score与自由空间冲突。
论文里的 mIoU、PQ、R@K 与 AP 各按原协议复现，不混合排名。

## 本轮保留候选 MICA-VSA

soft_mutual_association.py在当前深度可见的实测表面上计算双向覆盖F1，
与SBERT身份软分数、投影和颜色共同排序；保留原资格和一对一关联。
mutual_association.py保留第一轮硬门槛及共有可见性工具，以便解释失败，
不挂生产默认。没有场景、类别、GT/节点编号或指定实例数量条件。

| 扫描 | AP25 | AP50 | 平均AP(.50–.90) |
| --- | --- | --- | --- |
| scene0011_00 | 71.41 | 70.44 | 40.72 |
| scene0015_00 | 94.24 | 66.03 | 41.74 |
| scene0019_00 | 51.96 | 35.80 | 27.73 |
| 三扫描合并 | 75.36 | 60.60 | 37.77 |

真实融合至发布重放125帧，源掩码/特征逐字节相同，新增识图调用与数据
训练为0。合并相对v19增加0.26/2.32/0.67个百分点，0011退步、0019不变。
最终候选收据results/mutual_r2_evaluation.json与full_replay_validation.json；
原子/嵌套分解、直接ESAM融合、硬拒绝与支持比例降分的失败合并在
results/negative_trials.json。不能将掩码适配消融称为完整建图。

replay_cached_scene.py是可迁移的Linux入口；使用已存在的正式v19
场景与同一输入清单，输出到另一个dataset/scene目录。可恢复中断：
先验证已复制源掩码/特征及请求指纹，再从融合重跑，不重复识图。
加--validate-only只检查输入/缓存，不建图也不写完成标记；已在真实四场景
192帧执行入口检查，收据results/replay_entry_validation.json。

    /home/lewisliu/miniconda3/envs/scannet-sg/bin/python \
      tools/research/replay_cached_scene.py \
      --manifest /path/to/selected/manifest.json \
      --source /path/to/partaware_v19/scannet/sceneXXXX_00 \
      --target /path/to/research_cycle/mutual_r2/scannet/sceneXXXX_00

默认v19注册未改。后续必须按同协议独立评价全部三个扫描，检查0011
质量排序/重复、0019源SAM上下叠放物体误合并及跨域结构范围。
实际RGB证据与独立GT投影在综合报告27.7；GT投影只在审计程序中生成。
几何互不重叠不自动证明身份不同，连通分量也无法可靠拆开源合并掩码。

最小可选IoU校准器为12→32→1含偏置449参数，尚未实现/训练；先冻结
大模型，用独立训练扫描监督质量，并将存在性分开。简单支持比例降分
已经失败，不能用它充当校准器或删除少视角真实物体。当前单训练扫描
不足以证明泛化；这不阻止零新增训练的算法继续迭代。

## 连续研究当前状态：SAGE-CORE-V（未切换正式版本）

当前正式默认仍为v19。SAGE使用已经训练的官方ESAM几何候选和已有GPT
图像身份，普通无标注网格/segs提供实测表面；不读aggregation，不能称
独立纯GPT超越ESAM、纯在线125帧建图或隐藏形状生成。
proposal_ensemble.py、core_ownership.py、source_agreement.py分别负责
跨源重复/评分、实测核心所有权、点云/图发布；source_agreement_core.py
保留仲裁前实际云/图对照。source_refined_core.py只接受签名正支持掩码。
observed_domain_guard.py保留未归属真实顶点，阻止最近邻从邻近物体借标签。
核心模块与生产注册分离；先前三轮源码不得覆盖后再声称旧结果可复现。

三轮各125帧完整融合至最终发布、独立评价已完成，最新统一策略结果：

| 开发扫描 | AP25 | AP50 | 平均AP(.50–.90) |
| --- | --- | --- | --- |
| scene0011_00 | 97.25 | 79.04 | 43.78 |
| scene0015_00 | 92.14 | 72.58 | 49.80 |
| scene0019_00 | 66.58 | 55.36 | 31.91 |
| 三扫描合并 | 88.50 | 71.32 | 43.79 |

375帧源mask/feature逐字节相同，新增API/数据训练0。全部扫描AP50/均值
超过官方ESAM，但0019 AP25落后12.04个百分点，不能称目标完成；0015
AP25较正式v19仍下降。结果与验证见results/sage_verified_core_evaluation.json
和sage_full_replay_validation.json，阶段成品位于datasets/scannet-sg-processed/
research_cycle/sage_verified_core。源图局部追加、直接peer排序、跨源一致性、
双向覆盖NMS均已否决；质量/存在不能混为一谈，不人工删节点。

010也是正式升级必要验收。相同100帧，冻结109GT严格框66.53/43.55%，
104共同观测点72.73/56.41/35.33%。同一一般算法必须取得010可复核收益，
不换统计口径、按场景挑策略或把ESAM弱迁移当作本项目已改进。
010没有ScanNet网格分段，先在签名RGB-D观测域筛选，再真实云/图发布
与严格框复核；掩码提案不算完整重跑。bounded_overlap.py是精确稀疏
int64 Jaccard，与原计数等价，不是调整指标的算法。

本次Windows执行适配器仍在共享工作区v20_work，ROOT依赖该布局；
不可直接复制进Linux tools目录后声称可独立执行。普通算法/Context回调
已在此目录保存，Windows驱动与源输入哈希见根AGENTS和结果收据。
run_sage_verified_service.sh三扫描退出0；010验收服务保存build/eval日志与
010_sage_domain.exit。没有completion不报AP，不重复启动已有进程。

六篇论文的固定源码口径片段已执行，整套论文未全部部署。
ThinkGraphs同输入官方几何后端变体（未启用API代理）与CDIS共享前端
稀疏输入后端变体已经完成三个扫描；不冒充作者全文复现。
CDIS统一适配为全部选帧、强度/关联各1，AP25/AP50/均值分别
68.88/33.06/19.78、75.16/52.58/31.01、57.14/35.00/20.71%。
基线论文数字、后端变体、接口检查严格分列，详见报告26.11。
