"""Portable training readiness check; does not launch or install a trainer."""
import argparse,json,platform,subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--train-list',type=Path,required=True)
    p.add_argument('--val-list',type=Path,required=True)
    p.add_argument('--esam-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    train=set(args.train_list.read_text().split());val=set(args.val_list.read_text().split())
    if train&val:raise ValueError('Training and validation scenes overlap')
    config=json.loads(Path(__file__).with_name('training_plan.json').read_text())
    prohibited=set(config['diagnostic_only_scenes'])
    if train&prohibited:raise ValueError('Diagnostic scenes entered training split')
    present={x.name for x in args.dataset_root.iterdir() if x.is_dir()}
    available=sorted(train&present)
    gpu={}
    try:
        import torch
        gpu=dict(torch_version=torch.__version__,CUDA_available=torch.cuda.is_available(),
            devices=[dict(name=torch.cuda.get_device_name(i),
                memory_GiB=torch.cuda.get_device_properties(i).total_memory/1024**3,
                compute_capability=list(torch.cuda.get_device_capability(i)))
                for i in range(torch.cuda.device_count())])
    except ImportError:gpu=dict(torch_available=False)
    try:
        import MinkowskiEngine as ME
        sparse=dict(import_available=True,version=ME.__version__,training_CUDA_backward_verified=False)
    except ImportError:sparse=dict(import_available=False,training_CUDA_backward_verified=False)
    record=dict(machine=platform.node(),python=platform.python_version(),GPU=gpu,sparse=sparse,
        official_train_scenes=len(train),official_val_scenes=len(val),
        available_train_scenes=available,missing_train_scenes=len(train-present),
        scene_level_disjoint=True,diagnostic_scenes_not_in_train=True,
        official_ESAM_training_entry_exists=(args.esam_root/'tools/train.py').is_file(),
        official_online_config_exists=(args.esam_root/'configs/ESAM_CA/ESAM_online_scannet200_CA.py').is_file(),
        dataset_training_started=False,production_profile_changed=False,
        readiness='needs training data coverage and CUDA sparse backward validation')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
if __name__=='__main__':main()
