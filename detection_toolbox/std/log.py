from datetime import datetime

__all__ = ["dprint"]

'''
Adds a timestamp
'''
def dprint(*args):
    now = datetime.now()
    print('[{:02d}:{:02d}:{:02d}]: {}'.format(now.hour, now.minute, now.second, " ".join(map(lambda s: str(s), args))))