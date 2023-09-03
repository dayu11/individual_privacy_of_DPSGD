import torch
import torch.nn as nn

import numpy as np

import time

import sys

import threading
from multiprocessing import Pool
import os

from rdp_accountant import compute_rdp, get_privacy_spent

def computeEpsFunc(args):
    #get_privacy_spent(orders, rdp, target_delta=delta)
    pid, rdps, orders, delta, result, job_idx = args

    job_size = job_idx.shape[0]

    base_progress = max(job_size // 10, 1)

    for i, idx in enumerate(job_idx):

        eps, _, opt_order = get_privacy_spent(orders, rdps[idx], target_delta=delta)
        result[idx] = eps
        if(i%base_progress==0):
            progress = (i+1)/job_size
            #print('computing eps thread %d, progress: %.1f%%'%(pid, progress*100))

    return job_idx, result[job_idx]

def computeRdpFunc(args):
    pid, sigmas, all_q, result, job_idx, orders = args

    job_size = job_idx.shape[0]

    base_progress = max(job_size // 10, 1)



    for i, idx in enumerate(job_idx):
        q = all_q[idx]
        sigma = sigmas[idx]
        
        n_orders = orders.shape[0]

        result_ = compute_rdp(q, sigma, 1, orders=orders)

        result[idx] = result_
        if(i%base_progress==0):
            progress = (i+1)/job_size
            #print('comput rdp thread %d, progress: %.1f%%'%(pid, progress*100))

    return job_idx, result[job_idx]

class PrivacyLossTracker(nn.Module):
    def __init__(self, n, batchsize, sigma, init_norm=10, orders=np.arange(2, 1024, 1), delta=1e-5, rounding=0.1):
        
        self.init_norm = init_norm

        self.norms = torch.zeros(n).cuda() + init_norm
        
        self.rounding = rounding
        self.all_possible_norms = []
        tmp_norm = rounding
        while tmp_norm < init_norm:
            self.all_possible_norms.append(tmp_norm)
            tmp_norm += rounding

        self.all_possible_norms = torch.tensor(self.all_possible_norms).cuda()


        self.sigma = sigma
        self.batchsize = batchsize
        
        self.n = n
        self.orig_n = self.n
        self.q = batchsize/n

        # sampling probabilities for all data points
        self.all_q = np.array([self.q]*self.all_possible_norms.shape[0])

        self.delta = delta

        self.orders = orders

        init_rdp = compute_rdp(self.q, sigma, 1, orders=orders)
        
        self.accmulated_rdp = torch.zeros(size=(n, orders.shape[0])).cuda()

        self.current_rdp = torch.zeros(size=(n, orders.shape[0])).cuda() + torch.tensor(init_rdp).cuda().float()

        self.all_levels_rdp = torch.zeros(size=(self.all_possible_norms.shape[0], orders.shape[0])).cuda()




    
    def round_norms(self, idx): ## always need to call this when the norms or rdps are updated

        norm_diff = torch.abs(self.norms[idx].view(idx.shape[0], 1) - self.all_possible_norms)
        min_diff_idx = torch.argmin(norm_diff, dim=1)
        for i in range(idx.shape[0]):
            self.norms[idx[i]] = self.all_possible_norms[min_diff_idx[i]]
            self.current_rdp[idx[i]] = self.all_levels_rdp[min_diff_idx[i]]

    def update_sigma(self, sigma):
        self.sigma = sigma
        self.update_rdp()

    def get_avg_norm(self):
        return torch.mean(self.norms).item()

    def update_rdp(self):
        different_sigmas = {}

        for i, norm in enumerate(self.all_possible_norms):
            relative_sigma = self.sigma * (self.init_norm/norm).item()
            if(relative_sigma not in different_sigmas.keys()):
                different_sigmas[relative_sigma] = [i]
            else:
                different_sigmas[relative_sigma].append(i)

        sigmas_to_compute = list(different_sigmas.keys())

        
        full_job_size = len(sigmas_to_compute)
        full_idx = np.arange(full_job_size)

        num_workers = os.cpu_count() // 2
        if(full_job_size<20):
            num_workers = 1

        per_workder_load = full_job_size // num_workers

        job_idxs = []

        for i in range(num_workers):
            if(i == num_workers - 1):
                idx = full_idx[i*per_workder_load:]
            else:
                idx = full_idx[i*per_workder_load:(i+1)*per_workder_load]
            job_idxs.append(idx)
        result = np.zeros(shape=(full_job_size, self.orders.shape[0]))

        args_list = []
        for i in range(num_workers):
            #pid, sigmas, q, result, job_idx, orders
            args = [i, sigmas_to_compute, self.all_q, result, job_idxs[i], self.orders]
            args_list.append(args)

        with Pool(num_workers) as p:
            result_tuples = p.map(computeRdpFunc, args_list)
            res_list = []
            for tup in result_tuples:
                res_list.append(tup[1])


            result = np.concatenate(res_list)


        for i in range(full_job_size):
            sigma = sigmas_to_compute[i]
            rdp = result[i]

            idx = different_sigmas[sigma]

            self.all_levels_rdp[idx] = torch.tensor(rdp).cuda().float()

        self.round_norms(np.arange(self.n))


    def update_norm(self, norms, idx):
        self.norms[idx] = norms
        self.norms[self.norms>self.init_norm] = self.init_norm

        self.round_norms(idx)


    def update_loss(self):
        self.accmulated_rdp += self.current_rdp

    def parallel_get_eps(self):

        full_job_size = self.orig_n
        full_idx = np.arange(full_job_size)

        num_workers = os.cpu_count() // 2
        per_workder_load = full_job_size // num_workers

        job_idxs = []

        for i in range(num_workers):
            if(i == num_workers - 1):
                idx = full_idx[i*per_workder_load:]
            else:
                idx = full_idx[i*per_workder_load:(i+1)*per_workder_load]
            job_idxs.append(idx)

        result = np.zeros(full_job_size)

        args_list = []
        for i in range(num_workers):
            #pid, rdps, orders, delta, result, job_idx = args
            args = [i, self.accmulated_rdp.cpu().numpy(), self.orders, self.delta, result, job_idxs[i]]
            args_list.append(args)

        with Pool(num_workers) as p:
            result_tuples = p.map(computeEpsFunc, args_list)
            res_list = []
            for tup in result_tuples:
                res_list.append(tup[1])
            
            result = np.concatenate(res_list)

        return result
