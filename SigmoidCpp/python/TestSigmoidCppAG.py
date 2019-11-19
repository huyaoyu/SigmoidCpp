import torch
from SigmoidCppAG import SigmoidCppM

if __name__ == "__main__":
    dc = SigmoidCppM()

    x = torch.rand((2,2), requires_grad=True)  # The input data.
    Y = torch.rand((2,2), requires_grad=False) # The true data.

    # Forward.
    y = dc(x)

    # Compute the loss.
    L = Y - y

    # Backward.
    L.backward(torch.ones(2,2))

    print("x = {}. ".format(x))
    print("y = {}. ".format(y))
    print("x.grad = {}. ".format( x.grad ))

    # Manually compute the radient.
    pLpx = -1.0 * (1.0 - y) * y

    print("PartialLPartialX = {}. ".format(pLpx))