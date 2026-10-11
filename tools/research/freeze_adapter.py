"""Freeze selected pretrained modules, including mutable norm statistics."""
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class EviSceneFreezeHook(Hook):
    def __init__(self,prefixes=('backbone','img_backbone')):
        self.prefixes=tuple(prefixes)

    def _modules(self,runner):
        model=getattr(runner.model,'module',runner.model)
        selected=[]
        for name in self.prefixes:
            module=getattr(model,name,None)
            if module is not None:selected.append(module)
        if not selected:raise ValueError('No declared pretrained module to freeze')
        return selected

    def before_train(self,runner):
        for module in self._modules(runner):
            module.requires_grad_(False)
            module.eval()

    def before_train_iter(self,runner,batch_idx,data_batch=None):
        # runner.model.train() must not re-enable frozen norm buffer updates.
        for module in self._modules(runner):module.eval()
