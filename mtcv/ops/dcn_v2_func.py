from torch.autograd import Function

from ._ext import dcn_v2 as _backend


# deprecated method. only worked for torch version <=1.1

class DCNv2Function(Function):
    def __init__(self, stride, padding, dilation=1, deformable_groups=1):
        super(DCNv2Function, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups


    def forward(self, input, offset, mask, weight, bias):
        if not input.is_cuda:
            raise NotImplementedError
        if weight.requires_grad or mask.requires_grad or offset.requires_grad or input.requires_grad:
            self.save_for_backward(input, offset, mask, weight, bias)
        output = input.new(*self._infer_shape(input, weight))
        self._bufs = [input.new(), input.new()]



class DCNv2PoolingFunction(Function):
