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

def get_directory(file_path, mode='name'):

    if sys.platform.startswith('linux'):
        path_pieces = file_path.split('/')
        sep_string = '/'
    else:
        path_pieces = file_path.split("\\")
        sep_string = "\\"

    if mode == 'name':
        return path_pieces[-2]
    elif mode == 'path':
        fragments = path_pieces[:-1]
        retPath = ""
        for f in fragments:
            retPath += f + sep_string + " "
        return retPath

def get_filename(file_path):

    if sys.platform.startswith('linux'):
        path_pieces = file_path.split('/')
        sep_string = '/'
    else:
        path_pieces = file_path.split("\\")
        sep_string = "\\"

    return path_pieces[-1]

def get_platform_path_custom(volume, path):

    if sys.platform.startswith('linux'):
        path = path.replace("\\", '/')
        path_pieces = path.split('/')
        sep_string = '/'
        volume = "/mnt/" + volume.lower()
    else:
        path = path.replace('/', "\\")
        path_pieces = path.split("\\")
        sep_string = "\\"
        volume = volume.upper() +  ":"

    retString = volume + sep_string
    for p in path_pieces:
           retString += p + sep_string
    retString = retString.rstrip(sep_string)

    return retString



if __name__ == '__main__':
    print(get_platform_path_custom('C', r"\Users\theko\Documents\Dataset\042000002\c.mp3 "))