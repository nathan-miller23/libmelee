import sys
import threading
from time import sleep
import _thread as thread

class TimeoutException(Exception):
    pass

def quit_function(fn_name):
    # print to stderr, unbuffered in Python 2.
    print('{0} took too long'.format(fn_name), file=sys.stderr)
    sys.stderr.flush() # Python 3 stderr is likely buffered.
    raise TimeoutException("timeout!")

class Timeout(object):
    def __init__(self, f, seconds):
        self.func = f
        self.s = seconds

    def __call__(self, *args, **kwargs):
        timer = threading.Timer(self.s, quit_function, args=[self.func.__name__])
        timer.start()
        try:
            self.func(*args, **kwargs)
            return True
        except TimeoutException:
            return False
        finally:
            timer.cancel()

    def __get__(self, instance, owner):
        from functools import partial
        return partial(self.__call__, instance)