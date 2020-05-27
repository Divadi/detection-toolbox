from datetime import datetime

__all__ = ["dprint"]

'''
Adds a timestamp
'''
def dprint(s):
    now = datetime.now()
    print('[{:02d}:{:02d}:{:02d}]: {}'.format(now.hour, now.minute, now.second, s))