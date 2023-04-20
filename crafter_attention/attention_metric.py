import os
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class AttnMetric:
    def __init__(self, objsize, path2attnmaps, path2slotmasks, path2envmaps, path2envinfo):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.objsize = objsize
        with open(path2attnmaps, "rb") as file:
            self.attnmaps = pickle.load(file) # (892*1*8*81)
        with open(path2slotmasks, "rb") as file:
            self.slotmasks = pickle.load(file) # (892*8*72*72)
        with open(path2envmaps, "rb") as file:
            self.envmaps = pickle.load(file)
        with open(path2envinfo, "rb") as file:
            self.envinfo = pickle.load(file)

        self.imgsize = self.slotmasks.shape[-1]
        self.nslots = self.slotmasks.shape[1]
        self.episodes_start = np.cumsum([0] + self.envinfo["episode_lengths"])
        self.episodes_rewards= self.envinfo["episode_rewards"]
     

    def binaryAttnMetric(self, attnmaps, envmaps, thres = 1e-1, is_object_agnostic = True):
        """
            Input: 
                attnmaps: slot based attention maps for the given episode
                envmaps: environment observations for the given epsisode
                thres: threshold for converting attnmaps to binary version
                is_object_agnostic:
                    if True: calculate total number of configurations, regardless of underlying environment layout
                    if False: calcualte the number of configurations per object
            Return:
                if is_object_agnostic:
                    the density of possible binary configurations per frame, per attention slot
                    the density of possible binary configurations per attention slot
                    the density of possible binary configurations
                if not is_object_agnostic:
                    the density of possible binary configurations per object, per frame, per attention slot
        """
        binaryslotattn = self.binaryMap(attnmaps, thres = thres) # len_eps * n_slots * imgsize * imgsize
        h_start = 0
        len_eps = binaryslotattn.shape[0]
        possible_configs, n_configs = self.countAttnConfigBinary(binaryslotattn, thres = thres, is_binary = True)
        # possible_objects, n_object = self.countObject(envobs)
        print("total number of configurtions:", n_configs)

        if is_object_agnostic:
            attn_configs = np.zeros((len_eps, self.nslots, n_configs)) # dictionary of possible attention configurations, total number of configurations: 2**(self.imgsize**2)
            for i in range(attnmaps.shape[0]):
                for slot_count in range(self.nslots):
                    for h_start in np.arange(0, self.imgsize, self.objsize):
                        for w_start in np.arange(0, self.imgsize, self.objsize):
                            curr_attnmap = binaryslotattn[i, slot_count, h_start:h_start+self.objsize, w_start:w_start+self.objsize]
                            binary_curr_attnmap = ''.join([str(num) for num in curr_attnmap.cpu().numpy().reshape(-1)])
                            decimal_curr_attnmap = int(binary_curr_attnmap, 2)
                            index = possible_configs.index(decimal_curr_attnmap)
                            attn_configs[i, slot_count, index] += 1
            return attn_configs, np.sum(attn_configs, axis = 0), np.sum(attn_configs.reshape[-1])
        else:
            pass
            

    def countAttnConfigBinary(self, attnmaps, thres = 0.1, is_binary = False):
        """
            for given attnmaps, return all possible configurations in decimal format and the number of possible configurations
        """
        possible_configs = []
        if not is_binary:
            binaryslotattn = self.binaryMap(attnmaps, thres = thres)
        for i in range(attnmaps.shape[0]):
            print(i)
            for slot_count in range(self.nslots):
                for h_start in np.arange(0, self.imgsize, self.objsize):
                    for w_start in np.arange(0, self.imgsize, self.objsize):
                        curr_attnmap = binaryslotattn[i, slot_count, h_start:h_start+self.objsize, w_start:w_start+self.objsize]
                        # print(attnmaps[i, slot_count, h_start:h_start+self.objsize, w_start:w_start+self.objsize])
                        # print(curr_attnmap)
                        binary_curr_attnmap = ''.join([str(num) for num in curr_attnmap.cpu().numpy().reshape(-1)])
                        decimal_curr_attnmap = int(binary_curr_attnmap, 2)
                        # print(binary_curr_attnmap)
                        # print(decimal_curr_attnmap)
                        if decimal_curr_attnmap not in possible_configs:
                            possible_configs.append(decimal_curr_attnmap)
        print("total number of configurations:", len(possible_configs))
        return possible_configs, len(possible_configs)
            

    def ObjectDict(self):
        # for given envmaps, the exhausitive list of objects and their representations
        # return all possible object reps and the number of all possible objects
        obj_list = []
        # for i in range(self.envmaps.shape[0]):
        for i in range(1):
            for h_start in np.arange(0, self.imgsize, self.objsize):
                for w_start in np.arange(0, self.imgsize, self.objsize):
                    curr_envmap = self.envmaps[i,:,h_start:h_start+self.objsize,w_start:w_start+self.objsize]
                    if len(obj_list) == 0:
                        obj_list.append(curr_envmap)
                    
                    new_append = False
                    is_break = False
                    for count, arr in enumerate(obj_list):
                        if np.array_equal(arr.cpu().numpy(), curr_envmap.cpu().numpy()):
                            if i==1: print("there is an equal")
                            is_break = True
                            break
                    if count == len(obj_list)-1 and not is_break:
                        new_append = True
                    if new_append:
                        if i == 1:
                            print("there is an append with h_start %d, w_start %d" % (h_start, w_start))
                        obj_list.append(curr_envmap)

        for i in range(1,2):
            for h_start in [0]:
                for w_start in [0]:
                    curr_envmap = self.envmaps[i,:,h_start:h_start+self.objsize,w_start:w_start+self.objsize]
                    if len(obj_list) == 0:
                        obj_list.append(curr_envmap)
                    
                    new_append = False
                    is_break = False
                    for count, arr in enumerate(obj_list):
                        if np.array_equal(arr.cpu().numpy(), curr_envmap.cpu().numpy()):
                            if i==1: print("there is an equal")
                            is_break = True
                            break
                    if count == len(obj_list)-1 and not is_break:
                        new_append = True
                    if new_append:
                        if i == 1:
                            print("there is an append with h_start %d, w_start %d" % (h_start, w_start))
                        obj_list.append(curr_envmap)
                        
                        
        return obj_list, len(obj_list)

    def countObject(self, envmaps):
        """
            Input:
                environment maps 
            Output:
                object identity at each location
        """
        # I need the exhausitive list of objects and their representations
        # I need to find the correspoding object identity for the given envmap
        pass

    def plotAttnDist(self, attnmaps, savepath = None):
        plt.hist(attnmaps.cpu().numpy().reshape(-1), bins=30, density=True, alpha=0.6, color="b")  # Use 30 bins, normalize to density, set transparency and color

        # Add labels and title
        plt.xlabel('Attention Intensity')
        plt.ylabel('Frequency')
        plt.title('Histogram of Attention Intensity')

        plt.savefig(savepath)

        
    def binaryMap(self, attnmaps, thres = 1e-1):
        """return binary version of the attnmap cutoff by the given threshold"""
        return (attnmaps > thres).to(torch.int)



path2attnmaps = "./debug_attn_patch_8_stride_8/kind-sky-14/valid_det/attn_maps_5"
path2slotmasks = "./debug_attn_patch_8_stride_8/kind-sky-14/valid_det/slot_masks_5"
path2envmaps = "./debug_attn_patch_8_stride_8/kind-sky-14/valid_det/episode_observations_5"
path2envinfo = "./debug_attn_patch_8_stride_8/kind-sky-14/valid_det/episode_details_5"

metric_cal = AttnMetric(objsize=8, path2attnmaps = path2attnmaps, path2slotmasks = path2slotmasks,
                        path2envmaps = path2envmaps, path2envinfo = path2envinfo)

#  visualization of the environment map
for i in range(10):
    plt.figure()
    plt.imshow(metric_cal.envmaps[i,:,:8,:8].permute(1,2,0).cpu().numpy())
    plt.savefig("/home/xuan/projects/def-bashivan/xuan/crafter-ood/crafter_attention/figures/" + "%d.png" % i)

# check object pixel level representation
print(metric_cal.envmaps[0, 0, 0:8, 0:8].cpu().numpy())
print(metric_cal.envmaps[0, 1, 0:8, 0:8].cpu().numpy())
print(np.array_equal(metric_cal.envmaps[0, :, 0:8, 0:8].cpu().numpy(), metric_cal.envmaps[1, :, 0:8,0:8].cpu().numpy()))
# print(np.array_equal(metric_cal.envmaps[0, :, 8*1:8*2,8*5:8*6].cpu().numpy(), metric_cal.envmaps[0, :, 8*1:8*2,8*6:8*7].cpu().numpy()))


# thres = 0.1
# plot the distribution of attention intensity
# metric_cal.plotAttnDist(metric_cal.slotmasks, savepath = "/home/xuan/projects/def-bashivan/xuan/crafter-ood/crafter_attention/figures/debug_attndist.jpeg")

# count the number of possible object and their corresponding representations
# obj_list, l = metric_cal.ObjectDict()
# print("number of objects:", l)
# for i, obj in enumerate(obj_list[:20]):
#     plt.figure()
    
#     plt.imshow(obj.permute(1,2,0).cpu().numpy())
#     plt.savefig("/home/xuan/projects/def-bashivan/xuan/crafter-ood/crafter_attention/figures/" + "%d.png" % i)
# print(np.array_equal(obj_list[0].permute(1,2,0).cpu().numpy(),obj_list[1].permute(1,2,0).cpu().numpy()))

# count the total number of configurations for given episode number
# for eps_count in range(len(metric_cal.episodes_start)):
#     _, l = metric_cal.countAttnConfigBinary(attnmaps = metric_cal.slotmasks[metric_cal.episodes_start[eps_count]:metric_cal.episodes_start[eps_count+1],:,:], thres = thres)
#     print("%d possible configurations for episode %d" % (l, eps_count))
#     break

# print(metric_cal.attnmaps.shape)
# print(metric_cal.attnmaps.max())
# print(metric_cal.attnmaps.min())

# print(metric_cal.slotmasks.shape)
# print(metric_cal.envmaps.shape)
# print(metric_cal.envinfo)

