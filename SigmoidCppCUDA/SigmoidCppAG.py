import torch

# from torch.utils.cpp_extension import load

# SigmoidCppCUDA = load( 
#     name="SigmoidCppCUDA", 
#     sources=[ "SigmoidCpp.cpp", "SigmoidCpp_Kernel.cu"],
#     verbose=True
#  )

import SigmoidCppCUDA

class SigmoidCppFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        s = SigmoidCppCUDA.forward(x)
        ctx.save_for_backward( s[0] )
        return s[0]

    @staticmethod
    def backward(ctx, grad):
        sv = ctx.saved_variables

        output = SigmoidCppCUDA.backward( grad, sv[0] )

        return output[0]

class SigmoidCppM(torch.nn.Module):
    def __init__(self):
        super(SigmoidCppM, self).__init__()
    
    def forward(self, x):
        return SigmoidCppFunction.apply( x )
