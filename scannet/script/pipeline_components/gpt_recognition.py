"""Optional GPT category stage with a shared cache and unchanged tag JSON."""
from pathlib import Path


def recognize(context):
    if not context.manifest:
        raise ValueError('GPT recognition requires an existing ScanNet-SG input manifest')
    from vision_api import Settings, recognize_manifest, attach_cache
    settings = Settings.load(context.recognition_config)
    cache = context.recognition_cache or Path.home() / 'datasets/scannet-sg-processed/gpt_vision_cache'
    source = recognize_manifest(context.manifest, cache, settings)
    attach_cache(source, context.scene, context.manifest)
    context.event('GPT类别识别完成并绑定共用缓存', cache=str(source), model=settings.model,
                  qwen_api_calls=0, shared_across_versions=True)
