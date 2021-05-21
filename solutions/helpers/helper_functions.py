"""Helper functions for the jupyter notebooks"""

from os.path import abspath
from socket import gethostname, gethostbyname

REMOTE_HOST="192.168.11.3"

def translate_to_local_file_path(filename,directory=''):  
    if (gethostbyname(gethostname())) == REMOTE_HOST :
        if directory:
            filepath= f"../{directory}/{filename}"
        else:
            filepath= f"../{filename}"
    else:
        if directory:
            filepath= f"../{directory}/{filename}"
        else:
            filepath= f"../{filename}"
    print(abspath(filepath))   
    return f"file:///{abspath(filepath)}"

def translate_to_file_string(filepath):
    return f"file:///{abspath(filepath)}"
