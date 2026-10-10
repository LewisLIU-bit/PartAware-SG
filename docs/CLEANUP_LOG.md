# 主流程代码清理记录

日期：2026-10-05。按用户“已不再使用、不再在主流程中的可以删掉”的要求，删除 61 个闲置代码、示例或旧说明文件。删除前核对保留的 40 个 Python 文件的导入和脚本路径，未发现指向被删除 Python 模块的依赖。

保留 ScanNet RGB-D 准备、Hypersim manifest、RAM/Qwen、DINO/Florence/SAM、默认组件、原始 C++ 融合回退、空间图、点云精修、部件建图、独立评价、共用可视化和回归检查。`grounded_sam_simple_demo.py` 实际提供主分割器；`align_instances.py` 的函数被精修导入，因此均保留。第三方运行库、环境、权重、原始数据和已生成实验产物未列入清理。

`run_scannet_sg.sh` 原第 3、4 阶段的旧分支位于共享主流程调用及退出语句之后，永远不会执行；本轮删除这 2438 个字符的重复代码，保留原位置参数与 clean 恢复入口。原 C++ 回退仍由核心 Python runner 调用。

README 删除重复安装、旧 v1/v2 示例、已退休脚本和失效路径；安装文档修正交叉链接。用户已删除根目录 `RESEARCH_LOG.md`，本轮保留这一删除并修正相关链接。公式报告 `RESEARCH_LOG.html` 和可编辑 `RESEARCH_LOG.tex` 保留。

共用可视化原默认路径指向开发者旧 `/home/cc` 文件夹，已改为项目自带 sample_data 的相对路径；显式传入点云和 JSON 的接口不变。自带样例检查通过，载入 48 个物体。

验证：30 项回归检查通过；真实 ScanNet 0、3、6 三帧在原始/默认两条路线均生成 2 个物体；两套现有 v3 的正式物体—部件图接口检查通过。临时 ScanNet 检查产物自动清理。实际节点结果未因删除闲置脚本而重建或人工调整。

用户随后追加 ai_001_010 原始前端对照：现有 RAM++ 模块增加原生 4585 类词表入口，保持既有开放词表用法；现有验证工具加入 RAM 原生/Qwen 标签复用的 DINO/SAM → 原始 C++ 对照入口。这些是本轮实际使用的核心代码，不属于闲置文件。

两组对照已完整处理同一 100 帧，分别生成 50、46 个正式物体；Qwen 100 个类别文件逐文件哈希一致，未重新调用 API。两组共用可视化接口检查通过，AP25/AP50 和绝对自然对数数量误差与已有两条路线一起记录在 HTML 报告第 15 节。评价产物仍放在 datasets 的独立结果目录，未写入源码。

## 删除文件清单

- `download/Download_ScanNet_SG.md`
- `download/Terms of Use.pdf`
- `download/download_and_upzip.py`
- `download/zip_files.py`
- `download/zip_files_unpack.py`
- `package.xml`
- `scannet/readme_openset.md`
- `scannet/readme_scannet509.md`
- `scannet/readme_subscan.md`
- `scannet/script/add_pose_bbox_to_frame_json.py`
- `scannet/script/align_instances_for_all.py`
- `scannet/script/alignment_examine.py`
- `scannet/script/fix_name_csv.py`
- `scannet/script/frame_ptc_all.py`
- `scannet/script/generate_subscans.py`
- `scannet/script/generate_subscans_clean_json_not_used.py`
- `scannet/script/get_instance_names.py`
- `scannet/script/grounded_sam/grounded_sam/chatgpt_image.py`
- `scannet/script/grounded_sam/grounded_sam/chatgpt_text.py`
- `scannet/script/grounded_sam/grounded_sam/image_ros_server.py`
- `scannet/script/grounded_sam/grounded_sam/simple_client.py`
- `scannet/script/grounded_sam/grounded_sam/simple_server.py`
- `scannet/script/grounded_sam/scannet_process/decompress_sens.py`
- `scannet/script/grounded_sam/scannet_process/download-scannet.py`
- `scannet/script/grounded_sam/scannet_process/get_grounded_seg_features.py`
- `scannet/script/grounded_sam/scannet_process/show_colored_instance.py`
- `scannet/script/map_generator_all.py`
- `scannet/script/map_generator_openset_all.py`
- `scannet/script/matcher_data_generation.py`
- `scannet/script/matcher_data_subscan.py`
- `scannet/script/openai_tools/check_batch_status_and_retrieve.py`
- `scannet/script/openai_tools/check_uploaded_batches.py`
- `scannet/script/openai_tools/decode_batch_results.py`
- `scannet/script/openai_tools/readme.md`
- `scannet/script/openai_tools/scenes_inference_all.py`
- `scannet/script/openai_tools/submit_batch_to_openai.py`
- `scannet/script/openai_tools/write_batch_jsonl.py`
- `scannet/script/ram/extract_objects_from_csv.py`
- `scannet/script/ram/inference_ram_plus.py`
- `scannet/script/sequence_matcher_data_generation.py`
- `scannet/src/get_fused_object_features.cpp`
- `scannet/src/scannet_per_frame_points.cpp`
- `scannet/src/scannet_ply_map.cpp`
- `scannet/utils/data_analysis.py`
- `scannet/utils/depth_image_check.py`
- `scannet/utils/feature_comparison_test.py`
- `scannet/utils/scan_to_scan_same_scene_overlap_cal.py`
- `scannet/utils/scans_matched_ratio_rank.py`
- `scannet/utils/show_ply_unique_colors.py`
- `scannet/utils/unique_names_counting.py`
- `scannet/utils/visualize_sequence_matching.py`
- `script/random_map_generator.py`
- `script/read_map.py`
- `script/render_research_report.cjs`
- `script/utils/instance_seg_visualization.py`
- `script/utils/ply_bbox_viewer.py`
- `script/utils/ply_viewer.py`
- `script/utils/recolor_ply_with_id.py`
- `script/visualize_gt_subscan.py`
- `script/visualize_partaware.py`
- `src/read_and_visualize_map.cpp`

## 2026-10-08：旧接口与未采用试验清理

核对实际源码及忽略路径后，旧 GPT-4o 的 chatgpt_image、chatgpt_text 和 openai_tools 批处理代码已在前次清理中删除；本轮确认无活跃调用，移除废弃 GPT-4o/GPT-4.1-nano 输出目录忽略项。新的中转 GPT 接口位于 vision_api.py，不复活旧接口。

删除未注册、无调用者的两个失败原型：pipeline_components/sam_fine.py、gated_sam.py。失败算法的实验说明留在研究报告第19章；保留实际使用的 FOVEA、YOLOE-L、粗细实测证据隔离及回归检查。

SAM3访问被作者拒绝，用户决定放弃，删除 .env-sam3（7.6 GB）、thirdparty/SAM3 和仅含失败下载缓存的 checkpoints/sam3。未采用的 YOLOE-26 提示自由试验删除 .venv-fovea26 及 yoloe-26x-seg-pf.pt（185 MB）。移除两份过时 SAM3 操作文档及安装清单，最新环境入口改为 VISION_API.md。Hugging Face账号令牌、Windows/WSL已修复的网络配置、现行环境和公开YOLOE-v8-S/L权重保留。

本轮仍保留已真实验证但尚未获收益的通用补全核心，避免删除尚在研究的原始点云/图像联合适配；第三方源码和大权重作为本地运行资源忽略，不混入核心源码提交。清理前逐一核实递归目标的绝对实际路径在 PartAware-SG 内；没有修改数据输入、旧试验成品或当前GPT缓存。ScanNet/Hypersim、RAM原始对照、Qwen缓存、DINO/Florence/SAM、VLPart、原C++回退、独立评价和共用可视化保留。

## 2026-10-09：GPT 构建失败试验清理

保存图和点云哈希、已完成评价及中文处理日志后，删除第一场景13个未采用的中途构图目录：v11r、v11r2至v11r9、v12r及v12r2至v12r4。清理逐个核对绝对路径位于 scannet-sg-processed，并确认没有重解析链接；不按通配符批量删除。审核镜像为 outputs/gpt_rejected_trial_receipts.json 和 outputs/gpt_trial_logs。

保留已验收的 v11r10、v12r5、原始DINO对照、原18物体融合观测、两场景输入及共用GPT原始响应缓存。第二场景运行目录不动，成功识图图像不重复请求，模型依赖不做临时安装。源码新增独立可见归属组件及真实有界Florence评分，不保留失败外观融合试验的代码副本作为主流程实现。既有 Grounded-Segment-Anything 子模块更改保持原状。

## 2026-10-09：ModelScope SAM3隔离与默认版本整理

按本轮新授权从ModelScope公开仓库重新取得SAM3权重并校验平台SHA256，官方源码固定提交，依赖集中在WSL Conda的sg-sam3环境；基础scannet-sg环境保留。源码及权重作为忽略的本地模型资源，环境清单与来源审核作为核心文档纳入版本管理。官方源码压缩包解压校验后删除，不保留用于下载的源码副本。没有复活此前项目内.env-sam3环境。

报告默认来源固定为原始RAM、v11既有Qwen缓存、v12既有GPT缓存。停止GPT v11新试验，历史结果及v12依赖的完整共享观测仍保留。SAM3新产物分别放在两个_v12_sam3r1目录；完整实验未验收前不删除已验收旧产物、不将临时结果标为默认优胜结果。新增SAM3核心组件可通过代码导入/注册点撤除，不保留逐算法参数开关或备份代码作为拆卸方式。

## 2026-10-09：数据最终归并与对比报告精简

按用户最新要求，datasets只保留scannet、scannet-sg-input、scannet-sg-processed。删除3RScan/KITTI，不建立备份。processed保留19个最终目录：partaware_v1至partaware_v12、RAM原始、GPT原始、Qwen→DINO、共享GPT缓存及scene0802基础/PartAware/fuse。v12的一套最终包内包含两个Hypersim场景；此前临时目录已清理，前述旧目录保留状态由此条更新。

清理后processed由约11 GiB降至4.4 GiB。scene0802的443份矫正位姿核对一致，矫正输入及正常成品保留。199帧SAM3缓存转入各最终v12场景，逐文件哈希一致并实际载入验证；共享GPT响应不重复请求。中文算法审核、正常图/PLY、轨迹、部件及最终评价保留，无用调试日志和14份重复评价删除；最终指标均归并到evaluation.json，不改变分数。

综合报告只保留各版本最终结果，默认原始RAM、v11既有Qwen、v12既有GPT。GPT原始流程明确进入原始流程及v12对比，两个场景与v12的输入、GT及评价参数核对一致。四份当前正式图验证及178项离线回归通过。第二场景v12的全面不退步门控未通过，报告保留这一限制；没有以清理来隐藏最终假阳性。

完整清理与保留记录见datasets_cleanup.json；最终结果和文件哈希见sam3_woc_results.json。报告最新章节为18（v12）、19（原始流程对比）、20（低AP诊断）、21（数据和直接可视化命令）。

保留完整性补充核对：清理时误删v1依赖的hypersim_joint_v4共享点云。图、部件及已记录历史评分保留；尚未找到同SHA256副本，原点云可视化暂不可复现。没有使用其他版本点云或生成点替代。 原评价记录的点云SHA256为7b8044a7bacdf25d517eca61c8999ff72afff0611fb97da639a59efb463cdd5a；保留数据及项目点云中无一致副本。此限制已写入v1版本章节、evaluation.json和最终结果索引。历史评价若图哈希不同也按原记录保留，不伪造为重算结果；当前GPT原始及v12的图/点云哈希核验一致。

## 2026-10-10 v15 final retention

Retained one v15 final tree with both scenes and identical signed source caches. Removed seven own v15 trial cache directories and the unused, unregistered mask_separation experiment. All earlier official finals, original RAM/GPT/Qwen baselines, ScanNet/Hypersim/scene0802 and pre-existing source modifications remain. No backup was made. Comprehensive report includes final version results only.

## 2026-10-10 v16 final retention

Retained one v16 final tree with both scenes and 199 identical signed observations. Removed only the seven own v16 diagnostic/trial cache directories. Original RAM/GPT/Qwen baselines, previous final versions, ScanNet/Hypersim/scene0802 and pre-existing source modifications remain. No backup was made. Consolidated report shows final results only.


2026-10-10：更新v16唯一最终版后，清除本轮自己创建的MIRA/正交面/完整复跑候选缓存；未删除历史版本最终成品、ScanNet、Hypersim或scene0802。最终综合报告只保留各版本终极结果和真实未解决问题。
