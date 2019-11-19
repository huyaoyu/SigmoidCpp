import torch
from torch.autograd import Variable

from SigmoidCppAG import SigmoidCppM

if __name__ == "__main__":
    dc = SigmoidCppM()

    x = Variable( torch.rand((2,2)).cuda(), requires_grad=True )  # The input data.
    Y = Variable( torch.rand((2,2)).cuda(), requires_grad=False ) # The true data.

    # Forward.
    y = dc(x)

    # Compute the loss.
    L = Y - y

    # Backward.
    L.backward(torch.ones(2,2).cuda())

    print("x = {}. ".format(x))
    print("y = {}. ".format(y))
    print("x.grad = {}. ".format( x.grad ))

    # Manually compute the radient.
    pLpx = -1.0 * (1.0 - y) * y

    print("PartialLPartialX = {}. ".format(pLpx))