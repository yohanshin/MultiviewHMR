import torch
from torch.utils.tensorboard import SummaryWriter

import time
import copy
import os
import os.path as osp

class Timer():
    def __init__(self):
        self.init_time = time.time()
        self.curr_time = time.time()
    
    def __call__(self):
        delta_T = time.time() - self.init_time
        delta_t = time.time() - self.curr_time
        self._update_time()

        return delta_t, delta_T

    def _update_time(self):
        self.curr_time = time.time()


class Logger():
    def __init__(self, model_dir, log_dir, write_freq, exp_name, total_iteration):
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.write_freq = write_freq
        self.exp_name = exp_name
        self.total_iteration = total_iteration
        
        if not osp.exists(log_dir):
            os.makedirs(log_dir)

        if not osp.exists(model_dir):
            os.makedirs(model_dir)

        self.writer = SummaryWriter(
            log_dir=log_dir, 
            comment=exp_name
            )

        self.timer = Timer()
        self.read_results()
        
    def __call__(self, losses, iteration, is_val=False):
        if not is_val:
            iter_in_epoch = iteration % self.total_iteration
            if iter_in_epoch % self.write_freq == 0 and iter_in_epoch > 0:
                self.write_tensorboard(losses, iteration)
                self.print_loss_status(losses, iteration)
        
        else:
            epoch = iteration + 1
            self.write_tensorboard(losses, epoch)

    def get_total_loss(self, losses):
        total_loss = 0
        for l in losses:
            total_loss += losses[l]
        losses['total_loss'] = total_loss

        return losses
    
    def print_loss_status(self, losses, iteration):
        _epoch = iteration // self.total_iteration + 1
        _iteration = iteration % self.total_iteration

        delta_t, delta_T = self.timer()
        _hours = int(delta_T) // 3600
        _minutes = (int(delta_T) % 3600 ) // 60
        _seconds = int(delta_T) % 60

        output_msg = "Training ... Exp: %s | [Epoch: %d | iteration: %d/%d] | "%(self.exp_name, _epoch, _iteration, self.total_iteration)
        output_msg = output_msg + 'time: %.2f sec (overall: %02d:%02d:%02d sec) | Losses: '%(delta_t, _hours, _minutes, _seconds)
        for i, l in enumerate(losses):
            output_msg = output_msg + '{}. '.format(str(i+1)) + l + ' %.4f,   '%losses[l]
        
        print(output_msg, flush=True)

    def add_scalars(self, tag, losses, iteration):
        self.writer.add_scalars(tag, losses, iteration)

    def write_tensorboard(self, losses, iteration):
        for l in losses:
            self.writer.add_scalar(l, losses[l], iteration)

    def read_results(self):
        self.results = []
        result_file = osp.join(self.model_dir, 'checkpoint.txt')
        if osp.exists(result_file):
            f = open(result_file, 'r')
            results = f.readlines()

            for result in results:
                self.results.append(float(result.split(' ')[-1][:-1]))
        
            f.close()

    def save_last_statedict(self, model):
        last_statedict_pth = osp.join(self.model_dir, 'last_statedict.pt')
        torch.save(model.state_dict(), last_statedict_pth)

        return last_statedict_pth

    def save_checkpoint(self, model, optimizer, epoch, iteration, mpjpe, recone, mpjpe2=None, recone2=None):
        self.results.append(recone)
        
        if recone == min(self.results):
            checkpoint_path = osp.join(self.model_dir, "checkpoint.pt")
        else:
            checkpoint_path = osp.join(self.model_dir, "last_checkpoint.pt")
        checkpoint = ({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'iteration': iteration
        })
        torch.save(checkpoint, checkpoint_path)

        with open(osp.join(self.model_dir, 'checkpoint.txt'), 'a') as write_file:
            if recone2 is None:
                str_to_write = 'Epoch: %03d,   MPJPE: %.2f,   RECONE: %.2f'%(epoch+1, mpjpe, recone)
            else:
                str_to_write = 'Epoch: %03d,   H36M-MPJPE: %.2f,   H36M-RECONE: %.2f,   MPI-MPJPE: %.2f,   MPI-RECONE: %.2f'%(
                    epoch+1, mpjpe, recone, mpjpe2, recone2)
            write_file.writelines(str_to_write + '\n')
        

def build_logger(cfg, total_iteration):
    return Logger(model_dir=osp.join(cfg.TRAIN.LOGGER.OUTPUT_PTH, cfg.TRAIN.NAME),
                  exp_name=cfg.TRAIN.NAME,
                  log_dir=osp.join(cfg.TRAIN.LOGGER.OUTPUT_PTH, cfg.TRAIN.NAME, 'log'),
                  write_freq=cfg.TRAIN.LOGGER.WRITE_FREQ,
                  total_iteration=total_iteration,
                  )