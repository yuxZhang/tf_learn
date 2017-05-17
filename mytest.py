class test_smp(object):
    def  __init__(self, val):
        self.val = val

    def __enter__(self):
        print "In __enter__()"
        # return 1


    def __exit__(self, type, value, trace):
        print "In __exit__()"
        # return 1



def test_fun(val):
    return test_smp(val)

with test_fun(23) as a:
    pass