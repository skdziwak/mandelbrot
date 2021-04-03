import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from matplotlib import cm
from matplotlib import pyplot as plt
from PIL import Image
from pycuda.compiler import SourceModule
import shutil, os
import argparse
import random
import traceback