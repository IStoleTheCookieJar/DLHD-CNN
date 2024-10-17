# This will hopefully output the architecture for a specified arch type:

import numpy as np
from architecture import *

arch = 'hp3'
lr = 0.0011109555681977205
wd = 0.007054871539028646
dr = 0.35228908177827717
hidden = 6

get_arch = get_architecture(arch+'_err', hidden, dr, channels=1)

print(get_arch)
