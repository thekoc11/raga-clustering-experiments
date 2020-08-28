import sys
import os
FILE_PRE = ''

if sys.platform.startswith('linux'):
    FILE_PRE = r'/mnt/c/'
else:
    FILE_PRE = r"C:\\"

def get_platform_path(path):
    retPath = ''
    if sys.platform.startswith('linux'):
        retPath = FILE_PRE + path.replace("\\", "/")
    else:
        retPath = FILE_PRE + path.replace("/", "\\")

    return retPath
