import time

def tictoc(func):
    def wrapper(**kargs):
        t1 = time.time()
        func(**kargs)
        t2 = time.time() - t1
        print(f'{func.__name__} ran in {round(t2, ndigits=2)} seconds')
    return wrapper
