"""Record completed relational attempt without turning a screen into a map."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
work = ROOT/'v20_work'
receipt = json.loads((work/'relational_candidate/final_receipt.json').read_text())
assert receipt['complete'] and receipt['global_upgrade_rejected']
p = ROOT/'AGENTS.md'; text = p.read_text()
text = text.replace('当前保守归并14方法族，未训练；第15项尚在实现。',
                    '当前保守归并15方法族，数量前提达标；新训练尚未启动。')
text = text.replace('build_attempt_ledger.py将完整收据归并14个独立方法族；',
                    'build_attempt_ledger.py将完整收据归并15个独立方法族；')
text = text.replace('training_gate_passed=false，当前不开始训练，第15项不同方法尚待完整结果。',
                    'minimum_attempt_count_met=true；training_gate_passed=false不等于数量未达，\n  而是整体对照和独立训练/校准数据尚缺、未自动开启训练。')
if '### 15.13 第15项完成' not in text:
    text += '''

### 15.13 第15项完成、数量前提与下一条可执行路线

- 跨源共同可见身份关系：同一原帧中两可靠native对应不同mask，至少两个
  独立中心相距15cm；任意原raw whole同时覆盖二者则保持歧义，不当负关系。
  仅将覆盖一对分离身份各>=80%的peer暂定为复合几何，而不是单一物体。
  不删物理域点，不用GT/名称/预期数量作判定；原q/NMS/CORE和segs发布不变。
- 四域同规则构建/独立评价全部complete，build/evaluation/screen.exit=0；
  16项行为/未知/part/编号/点序/刚体/签名CORE等价测试通过，控制逐项复现。
  AP25/AP50/均值：0011=98.11/79.20/44.19%，0015=92.14/72.58/49.80%，
  0019=66.58/55.36/31.91%，010=43.22/26.87/13.92%。
  0011较控制+.85/+.16/+.42；0015/0019不变；010虽较CORE+.5.29/6.03/2.81，
  仍较v19低29.51/29.53/21.41。否决全域升级，没有完整新云/图或严格框成绩。
  API/训练/GPU=0，构建105秒/峰1353MiB，评价843MiB。
- 失败原因：四域只延期1/0/0/2个peer；0019的13对分离关系没有指向源mask
  已经合并的叠放椅子，因此不能凭关系凭空产生两个候选。010还留73/75个
  peer，部分错误覆盖和四个原生高IoU NMS抑制仍在；whole歧义保护部件，
  也可让真实混合保留。证据不足时保持未知，不按用户指出数量拆分。
- final_receipt、精确输入/每帧depth/pose/raw-mask/cache哈希、源码/测试在
  v20_work/relational_candidate，核心SHA ea3e471c0d489391be085f557092361e9cdaf98b65aa1c9c0142943944c8968a。
  作者MaskClustering源码固定eb2d41c2267cde21966f4340c37b8cf94b05c23c，
  只核读/借思想，没复现完整CropFormer+CLIP+图聚类系统。
- 保守15个方法族已完整；minimum_attempt_count_met=true，参数/重复/反事实
  不另计。不自动训练或说已穷尽算法；新五扫描我们的完整结果仍缺。
  可靠小训练建议：449参数12→32→1观测mask-IoU质量头，冻结大骨干，
  只用独立训练GT监督；几何质量低不等于不存在，未知不作不存在负例。
  此头无法创造0019缺失mask，不能以小头替代独立前端问题。
- 继续执行两项准备：cropformer_preflight核查独立冻结2D实体mask官方源码/
  权重/隔离环境与sm120兼容，针对源掩码一致误合并；quality_training_preflight
  核查0000训练数据与真实特征，准备449小头及梯度/划分契约。都是准备，
  不写为已部署/已训练或预测提升；不将八扫描/010GT训练再回测。
- 扩展并行提速：run_first_construction_service.sh等待0025完整共用缓存后
  立即跑原始/v19，其他缓存继续。run_scan200.py持完整事务construction.lock，
  后续suite检查最终marker后复用，GPU锁单独持有；不重复已有构建。
  构建前validate_recovered_cache.py严格查完整provenance、raw字节和恢复链，
  原生产validator未改；恢复缓存今后必须显式适配，不声称普通CLI兼容。
  当前状态查stream_recovery/first_construction进程、日志、.exit与complete。
- 同一heartbeat已更新检查新服务入口及读动态账本数量，不保留硬编码13。
  当前研究路径仍C盘/WSL，D盘只用于以后无项目任务。正式v19不变。
'''
p.write_text(text)
print('ATTEMPT15_HANDOFF_RECORDED',len(text.splitlines()),'lines')
