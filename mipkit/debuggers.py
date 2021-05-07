# ===============================================================================
# DEBUG
# ===============================================================================
class Debugger():
    def __init__(self):
        pass

    def set_trace(self):
        pass


def debug_pdb(is_exit=True):
    import pdb; pdb.set_trace();exit(1);
        

def debug_ipython(is_exit=True):
    import IPython; IPython.embed()
    if is_exit:
        exit(1)


