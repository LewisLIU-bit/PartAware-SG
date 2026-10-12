"""Refresh factual ongoing state without replacing the historical handoff."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
p = ROOT/'AGENTS.md'; text = p.read_text()
text = text.replace('当前保守归并13方法族，未训练。', '当前保守归并14方法族，未训练；第15项尚在实现。')
text = text.replace('共享GPT初始缓存正在续跑；ESAM预计算输入、五场景完整推理/独立评价已完成，',
                    '共享GPT初始缓存恢复中（已保留0025的33帧成功）；ESAM五场景完整推理/独立评价已完成，')
text = text.replace('build_attempt_ledger.py将完整收据归并13个独立方法族；',
                    'build_attempt_ledger.py将完整收据归并14个独立方法族；')
text = text.replace('training_gate_passed=false，当前不开始训练，至少再完成两项不同方法假设。',
                    'training_gate_passed=false，当前不开始训练，第15项不同方法尚待完整结果。')
start = text.index('- run_recovery_service.sh/resume_after_transport.py通过现有代理环境续跑，')
end = text.index('- 正式报告HTML四套核验通过，', start)
text = text[:start]+'''- 首轮恢复通过代理真实完成000720/000760后，在000920保存了failed流响应；
  resume.exit=1，不能说仍在运行。错误码upstream_http2_stream_error，
  随后001320又返回generic upstream_error；这是中转返回的上游状态，
  不证明OpenAI服务实际负载，不存在额度/账户原因的可靠证据。
- 显式恢复适配cached_relay_recovery.py不改冻结vision_api源码、模型、提示、
  输出预算或图像。只接受保存的failed上游流错误及原缺失终结事件，
  成功响应/拒绝/认证/额度失败不自动替换。原始失败按SHA归档；
  特定失败改非流transport且记录actual request/cache fingerprint与恢复链。
  两项离线测试验证只恢复失败、成功字节不变、归档完整及消费不新调用。
- run_stream_recovery_service.sh持独占恢复锁；外层最多2次，避免5次反复
  读取同一不可解析缓存。stream_recovery.log/.exit为当前状态入口，旧suite/
  resume失败留作历史。已保存0025的33成功帧，preflight逐帧哈希在扩展目录。
  续跑识图→原始/v19→已完成ESAM→统一CORE-V→独立评价，不能据启动填AP。
  不重跑已完成ESAM；失败请求是否计费未知，缓存初始调用不能写为零成本。
''' +text[end:]
if '### 15.12 CUE第14项' not in text:
    text += '''

### 15.12 CUE第14项与010独立因果审计

- CUE从独立原帧的可见mask身份给争议3D点分配正支持，仅唯一获得至少
  两个独立相机中心正身份支持时覆盖CORE；背景/未知中性，平票或无支持
  退回CORE。q、NMS、候选观测并集、ScanNet普通segs多数发布均保持不变。
  这不是调分或原图名称准入阈值变体；4项行为/刚体/编号/点序测试通过。
- 四域同一构建和独立原生评价已完成，退出0；CORE控制逐项复现签名成绩。
  AP25/AP50/均值：0011=98.11/82.91/44.48%，0015=91.93/72.58/49.58%，
  0019=66.58/55.36/32.45%，010=44.85/28.54/15.26%。
  0015较同CORE退步0.20/0/0.22个百分点，010较v19仍低27.87/27.87/20.07。
  因此否决全域替换，不因0011提高宣布完成。只是掩码screen，没有真实
  新云/图发布或010新严格框结果；零API/训练/GPU，构建峰值2.545GiB。
- 根因：0015的局部native候选抢走peer椅子70个正确覆盖顶点，最佳IoU
  .7713→.7347；正图像支持仍可偏向部件。0011柜体也有局部覆盖退步。
  010最佳IoU增75项降8项不等于AP恢复；错误的来源候选和身份竞争仍严重。
- 完整紧凑收据/精确已执行builder/测试在v20_work/cue_candidate；源码哈希
  c1754c0d3688306f6a193a5d048a1e3263e476fb40233cc1d770070562801f7f，
  final_receipt.json.complete=true，service.exit=0。只保留2.892MiB必要审计。
- 独立010 audit不修改构建，稀疏计数和冻结原生评价逐项一致，退出0、峰值
  1596MiB：仅加raw peer但不做所有权已使AP25/AP50/均值降至46.49/33.52/19.49。
  CORE从原正确归属移出269641点，其中260469给peer，27个原>50%IoU实例
  跌落；原图准入后仍有25个跌落。14个保留native实例从>=100点降至不足100。
  NMS还压掉四个高IoU原生候选；这些GT/节点只在审计记录，不进构建条件。
  独立反事实不是新算法尝试，不作为第15项凑数。audit_010.json可复现归因。
- 第15项正在研究跨源复合peer的关系身份准入：在>=2独立原帧内存在两个
  分开的原生mask身份时，跨两者的peer缺乏单物体证据；若原whole mask也
  同时覆盖两者则证据歧义、保留未知。借鉴MaskClustering的多帧mask-ID
  分布过滤underseg观察者，不宣称复现论文或cannot-link首次实现。
  现有生产instance_consensus已有native-native分离反证，本项须改变跨源
  复合身份模型选择，且用同一规则跨四域验收后才计为完成。当前未训练。
'''
p.write_text(text)
print('DETAILED_HANDOFF_REFRESHED',len(text.splitlines()),'lines')
