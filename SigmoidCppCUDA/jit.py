from torch.utils.cpp_extension import load

SigmoidCppCUDA = load( 
    name="SigmoidCppCUDA", 
    sources=[ "SigmoidCpp.cpp", "SigmoidCpp_Kernel.cu"],
    verbose=True
 )

help(SigmoidCppCUDA)
