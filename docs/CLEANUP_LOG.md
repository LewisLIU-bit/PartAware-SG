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
