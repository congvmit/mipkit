# ===============================================================================
# DEBUGGER
#
# References
# [1] https://stackoverflow.com/questions/16867347/step-by-step-debugging-with-ipython
# ===============================================================================
import IPython
import ipdb
import pdb


class Debugger():
    def __init__(self, method='ipython'):
        """Debugger with available methods: ipdb, pdb or ipython

        Args:
            method (str, optional): debugger method. Defaults to 'ipdb'.
        """
        self.tracer = self.init_tracer(method)

    def init_tracer(self, method):
        assert method in ['ipdb', 'pdb', 'ipython']
        if method == 'ipdb':
            return ipdb.set_trace
        elif method == 'pdb':
            return pdb.set_trace
        elif method == 'ipython':
            return IPython.embed

    def set_trace(self, method=None):
        if method is not None:
            self.tracer = self.init_tracer(method)
        return self.tracer


set_trace = Debugger().set_trace()
