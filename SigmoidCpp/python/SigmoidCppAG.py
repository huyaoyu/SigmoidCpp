import torch

import SigmoidCpp

class SigmoidCppFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        s = SigmoidCpp.forward(x)
        ctx.save_for_backward( s[0] )
        return s[0]

    @staticmethod
    def backward(ctx, grad):
        sv = ctx.saved_variables

        output = SigmoidCpp.backward( grad, sv[0] )

        return output[0]

class SigmoidCppM(torch.nn.Module):
    def __init__(self):
        super(SigmoidCppM, self).__init__()
    
    def forward(self, x):
        return SigmoidCppFunction.apply( x )
