#!/usr/bin/python3
from pynq import Overlay
from pynq import Xlnk
import numpy as np
import time
xlnk=Xlnk()

ol=Overlay("vpu_ddr4_100M.bit")
for i in ol.ip_dict:
    print(i);
ol.download();

#gpio=ol.axi_gpio_0
#gpio.write(0,0xF);
#
