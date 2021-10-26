import mipkit
mipkit.init_rich_traceback()


def add(x, y):
    # import mipkit;mipkit.debug.set_trace();exit();
    # from IPython.core.debugger import Pdb; Pdb().set_trace()
    return x + y


add(1, '2')
