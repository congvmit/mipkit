import pytorch

def get_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']