from keopscore.formulas.Operation import Broadcast
from keopscore.formulas.VectorizedScalarOp import VectorizedScalarOp
from keopscore.formulas.maths.Scalprod import Scalprod
from keopscore.formulas.maths.Sum import Sum
from keopscore.formulas.maths.Square import Square
from keopscore.formulas.variables.Zero import Zero
from keopscore.formulas.variables.IntCst import IntCst, IntCst_Impl
from keopscore.formulas.variables.RatCst import RatCst, RatCst_Impl
from keopscore.utils.misc_utils import KeOps_Error

##########################
######    Divide     #####
##########################


class Divide_Impl(VectorizedScalarOp):
    """the binary divide operation"""

    string_id = "Divide"
    print_spec = "/", "mid", 3
    linearity_type = "first"

    def ScalarOp(self, out, arg0, arg1):
        """returns the atomic piece of c++ code to evaluate the function on arg and return the result in out"""
        return f"{out.id} = {arg0.id} / {arg1.id};\n"

    #  \diff_V (A/B) = ((\diff_V A) * B - A * (\diff_V B)) / B^2
    def DiffT(self, v, gradin):
        fa, fb = self.children
        if fa.dim == 1 and fb.dim > 1:
            return fa.DiffT(v, Sum(gradin / fb)) - fb.DiffT(v, fa * gradin / Square(fb))
        elif fb.dim == 1 and fa.dim > 1:
            return (
                fa.DiffT(v, fb * gradin) - fb.DiffT(v, Scalprod(gradin, fa))
            ) / Square(fb)
        else:
            return fa.DiffT(v, gradin / fb) - fb.DiffT(v, fa * gradin / Square(fb))

    # parameters for testing the operation (optional)
    nargs = 2  # number of arguments

    @staticmethod
    def torch_op():
        """equivalent torch operation"""
        import torch

        return torch.div


# N.B. The following separate function should theoretically be implemented
# as a __new__ method of the previous class, but this can generate infinite recursion problems
def Divide(arg0, arg1):
    if isinstance(arg0, Zero):
        return Broadcast(arg0, arg1.dim)
    elif isinstance(arg1, Zero):
        KeOps_Error("division by zero")
    elif isinstance(arg1, IntCst_Impl):
        return RatCst(1, arg1.val) * arg0
    elif isinstance(arg1, RatCst_Impl):
        return RatCst(arg1.q, arg1.p) * arg0
    else:
        return Divide_Impl(arg0, arg1)
