import importlib.util
import sys
import sysconfig
from os.path import join, dirname, realpath

###############################################################
# Initialize some variables: the values may be redefined later

numpy_found = importlib.util.find_spec("numpy") is not None
torch_found = importlib.util.find_spec("torch") is not None

import keopscore
from keopscore.config import *

# Instantiating the keopscore.config main classes for pykeops
pykeops_cuda = cuda_config
pykeops_openmp = openmp_config
pykeops_base = config

get_build_folder = pykeops_base.get_build_folder
gpu_available = pykeops_cuda.get_use_cuda()


def pykeops_nvrtc_name(type="src"):
    basename = "pykeops_nvrtc"
    extension = ".cpp" if type == "src" else sysconfig.get_config_var("EXT_SUFFIX")
    return join(
        (
            join(dirname(realpath(__file__)), "common", "keops_io")
            if type == "src"
            else config.get_build_folder()
        ),
        basename + extension,
    )


def pykeops_cpp_name(tag="", extension=""):
    basename = "pykeops_cpp_"
    return join(
        config.get_build_folder(),
        basename + tag + extension,
    )


python_includes = "$({python3} -m pybind11 --includes)".format(python3=sys.executable)
