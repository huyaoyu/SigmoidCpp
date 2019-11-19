
import argparse

import torch
from torch.autograd import Variable

from SigmoidCppAG import SigmoidCppM

def parse_dims(s):
    """
    s is a string in the formate of "BxHxW.", 
    weher B, H, and W are positive integers.
    """

    n = s.split("x")

    if (len(n) != 3):
        raise Exception("len(n) = %d. " % (len(n)))

    return int(n[0]), int(n[1]), int(n[2])

if __name__ == "__main__":
    # Handle arguments.
    parser = argparse.ArgumentParser(description="Test SigmoidCppCUDA.")

    parser.add_argument("--dim", type=str, default="2x4x4", \
        help="The dimensions to be tested. Must be 3 integers separated by 'x'")

    parser.add_argument("--show-details", action="store_true", default=False, \
        help="Show the actual values of the tensors. ")
    
    args = parser.parse_args()

    # Get the test dimensions.
    B, H, W = parse_dims(args.dim)

    assert B > 0
    assert H > 0
    assert W > 0

    # Create the new SigmoidCppM layer.
    dc = SigmoidCppM()

    x = Variable( torch.rand((B, H, W)).cuda(), requires_grad=True )  # The input data.
    Y = Variable( torch.rand((B, H, W)).cuda(), requires_grad=False ) # The true data.

    # Forward.
    y = dc(x)

    # Compute the loss.
    L = Y - y

    # Backward.
    L.backward(torch.ones(B, H, W).cuda())

    if ( args.show_details ):
        print("x = {}. ".format(x))
        print("y = {}. ".format(y))
        print("x.grad = {}. ".format( x.grad ))

    # Manually compute the radient.
    pLpx = -1.0 * (1.0 - y) * y

    if ( args.show_details ):
        print("PartialLPartialX = {}. ".format(pLpx))

    # Compute the error.
    e = pLpx - x.grad
    print("torch.norm(e) = {}. ".format( torch.norm(e) ))