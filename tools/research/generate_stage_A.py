"""Materialize an actual MMEngine config for a new lab environment."""
import argparse,hashlib,json,math
from pathlib import Path
from mmengine.config import Config

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--esam-root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--profile',choices=['24GB','48GB','80GB'],required=True)
    p.add_argument('--data-root',type=Path,help='Converted scannet200-mv data, not raw scans')
    p.add_argument('--pilot-epochs',type=int,help='Optional short plumbing run, not a final comparison')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    root=args.esam_root.resolve()
    source=root/'configs/ESAM_CA/ESAM_online_scannet200_CA.py'
    cfg=Config.fromfile(str(source),import_custom_modules=False)
    plan=json.loads(Path(__file__).with_name('training_plan.json').read_text())
    profile=plan['hardware_profiles'][args.profile]
    if not args.checkpoint.is_file():raise FileNotFoundError(args.checkpoint)
    cfg.load_from=str(args.checkpoint.resolve())
    imports=list(cfg.get('custom_imports',{}).get('imports',[]))
    cfg.custom_imports=dict(imports=imports+['freeze_adapter'],allow_failed_imports=False)
    cfg.custom_hooks=list(cfg.get('custom_hooks',[]))+[
        dict(type='EviSceneFreezeHook',prefixes=['backbone','img_backbone'])]
    cfg.train_dataloader.batch_size=profile['initial_scene_batch']
    accumulation=math.ceil(profile['effective_batch_via_accumulation']/profile['initial_scene_batch'])
    cfg.optim_wrapper.accumulative_counts=accumulation
    def frames(value):
        if isinstance(value,dict):
            if 'num_frames' in value:value['num_frames']=profile['initial_frames']
            for item in value.values():frames(item)
        elif isinstance(value,(list,tuple)):
            for item in value:frames(item)
    frames(cfg.train_dataloader.dataset)
    if args.data_root:
        data_root=str(args.data_root.resolve())+'/'
        def data(value):
            if isinstance(value,dict):
                if 'data_root' in value:value['data_root']=data_root
                for item in value.values():data(item)
            elif isinstance(value,(list,tuple)):
                for item in value:data(item)
        cfg.data_root=data_root
        for key in ['train_dataloader','val_dataloader','test_dataloader']:
            data(cfg[key])
    if args.pilot_epochs is not None:
        if args.pilot_epochs<1:raise ValueError('Pilot epochs must be positive')
        cfg.train_cfg.max_epochs=args.pilot_epochs
        cfg.train_cfg.val_interval=1
        cfg.param_scheduler.end=args.pilot_epochs
    # Keep official optimizer, matching/losses, architecture and validation.
    # Effective scene batch is per device; DDP multiplies it by world size.
    cfg.work_dir=str(args.output.parent.resolve()/'stage_A_run')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    cfg.dump(str(args.output))
    loaded=Config.fromfile(str(args.output),import_custom_modules=False)
    assert loaded.model.type==cfg.model.type and loaded.train_dataloader.batch_size==profile['initial_scene_batch']
    assert loaded.optim_wrapper.accumulative_counts==accumulation
    assert 'oneformer3d' in loaded.custom_imports.imports
    receipt=dict(config=str(args.output.resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        profile=args.profile,scene_batch=profile['initial_scene_batch'],frames=profile['initial_frames'],
        optimizer_type=cfg.optim_wrapper.optimizer.type,lr=cfg.optim_wrapper.optimizer.lr,
        accumulative_counts=accumulation,
        effective_scene_batch_per_device=profile['initial_scene_batch']*accumulation,
        class_agnostic_instances=cfg.model.decoder.num_instance_classes==1,
        auxiliary_semantic_classes=cfg.model.decoder.num_semantic_classes,
        pilot_epochs=args.pilot_epochs,
        training_started=False,GPU_capacity_verified=False,
        PYTHONPATH_required=str(Path(__file__).resolve().parent),
        sparse_CUDA_training_validation_required=True)
    args.output.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
