import copy

from torch.nn.utils import clip_grad

# from ..fp16_utils import allreduce_grads, wrap_fp16_model
from .hook import HOOKS, Hook


@HOOKS.register_module()
class OptimizerHook(Hook):

    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)}, runner.outputs['num_samples'])
        runner.optimizer.step()

# @HOOKS.register_module()
# class Fp16OptimizerHook(OptimizerHook):
#     """FP16 optimizer hook.
#
#     The steps of fp16 optimizer is as follows.
#     1. Scale the loss value.
#     2. BP in the fp16 model.
#     2. Copy gradients from fp16 model to fp32 weights.
#     3. Update fp32 weights.
#     4. Copy updated parameters from fp32 weights to fp16 model.
#
#     Refer to https://arxiv.org/abs/1710.03740 for more details.
#
#     Args:
#         loss_scale (float): Scale factor multiplied with loss.
#     """
