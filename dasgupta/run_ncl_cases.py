#!/usr/bin/env python

"""

This script runs the ncl cases.
"""


import os
import sys
import re
import time
import string
import math
import numpy as np
from optparse import OptionParser

def main():

    usage_str = "%prog year case_list_file"
    parser = OptionParser(usage = usage_str)

    (options, args) = parser.parse_args()

    if len(args) < 2:
        parser.print_help()
        sys.exit(2)

    # Read in arguments
    year = args[0]
    caseListFile = args[1]
    header = True

    for line in file(caseListFile):
        if header:
            header = line.split(',')
            header = False
            continue
        tmp = line.split(',')
    #year = tmp[1][0:4]
        print year
    tmp2 = tmp[2].split(':')
    hour = int(tmp2[0])
        #hour = int(tmp[2][0:2])
        print hour
        index = 2*hour
        #a = addfile("/ScenarioB/Part1/WyomingWeaMod/WRF361/WRFV3/test/em_real/2014/wrfout_d03_2014-02-28_00:00:00.nc","r")
        #filename = "/ScenarioB/Part1/WyomingWeaMod/WRF361/WRFV3/test/em_real/%s/wrfout_d03_%s_00:00:00.nc" %(year,tmp[1])
        filename = "/ScenarioB/Part1/WyomingWeaMod/WRF361/WRFV3/test/em_real/%s/wrfout_d03_%s_00:00:00" %(year,tmp[1])
    if os.path.exists(filename):

        cmd = "ncl 'a=addfile(\"/ScenarioB/Part1/WyomingWeaMod/WRF361/WRFV3/test/em_real/%s/wrfout_d03_%s_00:00:00\",\"r\")' 'it=%d' skewtASCIIout.ncl" %(year,tmp[1], index)

            print cmd
            #sys.exit(1)
            os.system(cmd)
    else:
        print "%s does not exist" %filename
        #sys.exit(1)



if __name__ == "__main__":

    main()
