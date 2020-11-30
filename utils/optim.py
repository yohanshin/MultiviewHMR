import torch

class Scheduler():
    def __init__(self, optimizer, n_step, t_iter, init_lr_set, sz_step=5):
        self._optimizer = optimizer
        self.t_iter = t_iter
        self.init_lr_set = init_lr_set
        self.n_steps = n_step
        self.sz_step = sz_step

    def step(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def state_dict(self):
        return self._optimizer.state_dict()

    def _get_lr_scale(self):
        curr_epoch = self.n_steps // self.t_iter
        decay_pow = curr_epoch // self.sz_step
        
        return (0.5) ** decay_pow
    
    def _update_learning_rate(self):
        lr = []
        scale = self._get_lr_scale()
        for _lr in self.init_lr_set:
            lr.append(_lr * scale)
        
        self.n_steps += 1

        for idx, param_group in enumerate(self._optimizer.param_groups):
            param_group['lr'] = lr[idx]


def build_scheduler(cfg, optimizer, curr_iter, total_iter):
    scheduler = Scheduler(optimizer, curr_iter, total_iter,
                          (cfg.TRAIN.LR_DEFAULTS, 
                           cfg.TRAIN.LR_DEFAULTS,
                           cfg.TRAIN.LR_BACKBONE))
    
    return scheduler


def build_optimizer(cfg, model):
    
    optimizer = torch.optim.Adam(
        [{'params': model.volumetric_regressor.parameters()},
        {'params': model.aggregator.parameters(), 'lr': cfg.TRAIN.LR_DEFAULTS},
        {'params': model.backbone.parameters(), 'lr': cfg.TRAIN.LR_BACKBONE}],
        lr=cfg.TRAIN.LR_DEFAULTS, betas=cfg.TRAIN.BETAS)

    return optimizer