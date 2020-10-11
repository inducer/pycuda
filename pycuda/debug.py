import pycuda.driver

import sys
from optparse import OptionParser
from os.path import exists

pycuda.driver.set_debugging()

parser = OptionParser(usage="usage: %prog [options] SCRIPT-TO-RUN [SCRIPT-ARGUMENTS]")

parser.disable_interspersed_args()
options, args = parser.parse_args()

if len(args) < 1:
    parser.print_help()
    sys.exit(2)

mainpyfile = args[0]

if not exists(mainpyfile):
    print("Error:", mainpyfile, "does not exist")
    sys.exit(1)

sys.argv = args

exec(compile(open(mainpyfile).read(), mainpyfile, "exec"))
