import math
import torch
from torch import Tensor
import torch.optim._functional as F
from torch.optim.optimizer import Optimizer, required
from typing import List, Optional


def check_in(t, l):
    for i in l:
        if t is i:
            return True
    return False

def dot(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        kd_grad_buffer: List[Optional[Tensor]],
        kd_momentum_buffer: List[Optional[Tensor]],
        kd_params: List[Tensor],
        *,
        weight_decay: float,
        momentum: float,
        momentum_kd: float,
        lr: float,
        dampening: float):
    for i, param in enumerate(params):
        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)
        if momentum != 0:
            buf = momentum_buffer_list[i]
            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            elif check_in(param, kd_params):
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
            else:
                buf.mul_((momentum_kd + momentum) / 2.).add_(d_p, alpha=1 - dampening)
            d_p = buf
        # update params with task-loss grad
        param.add_(d_p, alpha=-lr)

    for i, (d_p, buf, p) in enumerate(zip(kd_grad_buffer, kd_momentum_buffer, kd_params)):
        # update params with kd-loss grad
        if buf is None:
            buf = torch.clone(d_p).detach()
            kd_momentum_buffer[i] = buf
        elif check_in(p, params):
            buf.mul_(momentum_kd).add_(d_p, alpha=1 - dampening)
        else:
            if weight_decay != 0:
                d_p = d_p.add(p, alpha=weight_decay)
            buf.mul_((momentum_kd + momentum) / 2.).add_(d_p, alpha=1 - dampening)
        p.add_(buf, alpha=-lr)


class DistillationOrientedTrainer(Optimizer):
    r"""
    Distillation-Oriented Trainer
    Usage:
        ...
        optimizer = DistillationOrientedTrainer()
        optimizer.zero_grad(set_to_none=True)
        loss_kd.backward(retain_graph=True)
        optimizer.step_kd() # get kd-grad and update kd-momentum
        optimizer.zero_grad(set_to_none=True)
        loss_task.backward()
        optimizer.step() # get task-grad and update tast-momentum, then update params.
        ...
    """

    def __init__(
        self, 
        params, 
        lr=required, 
        momentum=0, 
        momentum_kd=0,
        dampening=0,
        weight_decay=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if momentum_kd < 0.0:
            raise ValueError("Invalid momentum kd value: {}".format(momentum_kd))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, momentum_kd=momentum_kd, dampening=dampening,
                        weight_decay=weight_decay)
        self.kd_grad_buffer = []
        self.kd_grad_params = []
        self.kd_momentum_buffer = []
        super(DistillationOrientedTrainer, self).__init__(params, defaults)

    @torch.no_grad()
    def step_kd(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        assert len(self.param_groups) == 1, "Only implement for one-group params."
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_kd_buffer_list = []
            weight_decay = group['weight_decay']
            dampening = group['dampening']
            lr = group['lr']
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    state = self.state[p]
                    if 'momentum_kd_buffer' not in state:
                        momentum_kd_buffer_list.append(None)
                    else:
                        momentum_kd_buffer_list.append(state['momentum_kd_buffer'])
                    
        self.kd_momentum_buffer = momentum_kd_buffer_list
        self.kd_grad_buffer = d_p_list
        self.kd_grad_params = params_with_grad
        return loss        

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        assert len(self.param_groups) == 1, "Only implement for one-group params."
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            momentum_kd = group['momentum_kd']
            dampening = group['dampening']
            lr = group['lr']
            
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])
            dot(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                self.kd_grad_buffer,
                self.kd_momentum_buffer,
                self.kd_grad_params,
                weight_decay=weight_decay,
                momentum=momentum,
                momentum_kd=momentum_kd,
                lr=lr,
                dampening=dampening)
            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer
            for p, momentum_kd_buffer in zip(self.kd_grad_params, self.kd_momentum_buffer):
                state = self.state[p]
                state['momentum_kd_buffer'] = momentum_kd_buffer
            self.kd_grad_buffer = []
            self.kd_grad_params = []
            self.kd_momentum_buffer = []
        return loss
