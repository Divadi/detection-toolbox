import os

__all__ = ["makedirs"]

'''
Creates dir_path (and all intermediate directories)
Note that exist_ok=False => nonempty_ok=False
If: exist_ok, nonempty_ok
    False     _          : same as os.makedirs(dir_path, exist_ok=False)
    True      False      : dir_path can exist, but it must be empty
    True      True       : dir_path can exist, and it can be non-empty
'''
def makedirs(dir_path, exist_ok=False, nonempty_ok=False):
    if os.path.isdir(dir_path): #! exists already
        if not exist_ok:
            raise Exception("{} already exists".format(dir_path))
        else:
            if len(os.listdir(dir_path)) != 0: #! nonempty
                if not nonempty_ok:
                    raise Exception("{} is not empty".format(dir_path))
                else:
                    return dir_path
            else: #! exists, but is empty
                return dir_path
    else: #! does not exist
        os.makedirs(dir_path, exist_ok=exist_ok)
        return dir_path