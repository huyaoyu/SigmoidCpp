from torch.utils.cpp_extension import load

DispCorrCUDA = load( 
    name="DispCorrCUDA", 
    sources=[ "SigmoidCpp.cpp", "SigmoidCpp_Kernel.cu"]
 )

help(DispCorrCUDA)
