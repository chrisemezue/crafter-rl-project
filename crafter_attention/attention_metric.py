import os
import pickle
import torch
import json
import PIL
import numpy as np
import imageio
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity

import io


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


class AttnMetric:
    def __init__(self, objsize, path2attnmaps, path2slotmasks, path2envmaps, path2envinfo):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(self.device)
        self.objsize = objsize
        with open(path2attnmaps, "rb") as file:
            # self.attnmaps = torch.load(file, map_location=torch.device('cpu'))
            self.attnmaps = CPU_Unpickler(file).load()
            # self.attnmaps = pickle.load(file) # (892*1*8*81) => why is this one not equivalent to slotmasks?
        with open(path2slotmasks, "rb") as file:
            self.slotmasks = CPU_Unpickler(file).load()
            # self.slotmasks = torch.load(file, map_location=torch.device(self.device)) # (892*8*72*72) => (771*8*144*144) [fit environment size]
        with open(path2envmaps, "rb") as file:
            self.envmaps = CPU_Unpickler(file).load()
            # self.envmaps = torch.load(file, map_location=torch.device(self.device))
        with open(path2envinfo, "rb") as file:
            self.envinfo = CPU_Unpickler(file).load()
            # self.envinfo = torch.load(file, map_location=torch.device(self.device))

        self.imgsize = self.slotmasks.shape[-1]
        self.nslots = self.slotmasks.shape[1]
        self.episodes_start = np.cumsum([0] + self.envinfo["episode_lengths"])  # starting index for each episode
        self.episodes_rewards = self.envinfo["episode_rewards"]

    def binaryAttnMetric(self, attnmaps, envmaps, thres=1e-1, is_object_agnostic=True):
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
        binaryslotattn = self.binaryMap(attnmaps, thres=thres)  # len_eps * n_slots * imgsize * imgsize
        h_start = 0
        len_eps = binaryslotattn.shape[0]
        possible_configs, n_configs = self.countAttnConfigBinary(binaryslotattn, thres=thres, is_binary=True)
        # possible_objects, n_object = self.countObject(envobs)
        print("total number of configurtions:", n_configs)

        if is_object_agnostic:
            attn_configs = np.zeros((len_eps, self.nslots,
                                     n_configs))  # dictionary of possible attention configurations, total number of configurations: 2**(self.imgsize**2)
            for i in range(attnmaps.shape[0]):
                for slot_count in range(self.nslots):
                    for h_start in np.arange(0, self.imgsize, self.objsize):
                        for w_start in np.arange(0, self.imgsize, self.objsize):
                            curr_attnmap = binaryslotattn[i, slot_count, h_start:h_start + self.objsize,
                                           w_start:w_start + self.objsize]
                            binary_curr_attnmap = ''.join([str(num) for num in curr_attnmap.cpu().numpy().reshape(-1)])
                            decimal_curr_attnmap = int(binary_curr_attnmap, 2)
                            index = possible_configs.index(decimal_curr_attnmap)
                            attn_configs[i, slot_count, index] += 1
            return attn_configs, np.sum(attn_configs, axis=0), np.sum(attn_configs.reshape[-1])
        else:
            pass

    def countAttnConfigBinary(self, attnmaps, thres=0.1):
        """
            for given attnmaps, return all possible configurations in decimal format and the number of possible configurations
        """
        possible_configs = []

        binaryslotattn = self.binaryMap(attnmaps, thres=thres)
        for i in range(attnmaps.shape[0]):

            for slot_count in range(self.nslots):
                for h_start in np.arange(0, self.imgsize, self.objsize):
                    for w_start in np.arange(0, self.imgsize, self.objsize):
                        curr_attnmap = binaryslotattn[i, slot_count, h_start:h_start + self.objsize,
                                       w_start:w_start + self.objsize]
                        # print(attnmaps[i, slot_count, h_start:h_start+self.objsize, w_start:w_start+self.objsize])
                        # print(curr_attnmap)
                        binary_curr_attnmap = ''.join([str(num) for num in curr_attnmap.cpu().numpy().reshape(-1)])
                        decimal_curr_attnmap = int(binary_curr_attnmap, 2)
                        # print(binary_curr_attnmap)
                        # print(decimal_curr_attnmap)
                        if decimal_curr_attnmap not in possible_configs:
                            possible_configs.append(decimal_curr_attnmap)
        # print("total number of configurations:", len(possible_configs))
        return possible_configs, len(possible_configs)

    def countAttnConfigBinaryperObject(self, attnmaps, obj_index, frame_obj_maps, thres=0.1):
        """
            for given attnmaps, return all possible configurations in decimal format and the number of possible configurations
        """
        possible_configs = []
        binaryslotattn = self.binaryMap(attnmaps, thres=thres)
        for i in range(attnmaps.shape[0]):
            # print("loop over frame %d" % i)

            for h_start in np.arange(0, self.imgsize, self.objsize):
                for w_start in np.arange(0, self.imgsize, self.objsize):
                    if frame_obj_maps[i, int(h_start / self.objsize), int(w_start / self.objsize)] != obj_index:
                        continue
                    assert frame_obj_maps[i, int(h_start / self.objsize), int(w_start / self.objsize)] == obj_index

                    for slot_count in range(self.nslots):
                        curr_attnmap = binaryslotattn[i, slot_count, h_start:h_start + self.objsize,
                                       w_start:w_start + self.objsize]  # do not have to consider about stride vs patch size here; but may have to consider perimeter regions of the attention region
                        binary_curr_attnmap = ''.join([str(num) for num in curr_attnmap.cpu().numpy().reshape(-1)])
                        decimal_curr_attnmap = int(binary_curr_attnmap, 2)

                        # identify the location of current object index in the config_dictionary
                        # todo: did not consider the density of each configuration

                        if decimal_curr_attnmap not in possible_configs:
                            possible_configs.append(decimal_curr_attnmap)

        return possible_configs

    ####### to be updated
    def ContinuousAttentionperObject(self, attnmaps, obj, obj_thres=1e-2):
        configs = []
        # print("attention map shape:",attnmaps.shape[0])
        for i in range(attnmaps.shape[0]):
            # print("frame %d in processing" % i)
            for h_start in np.arange(0, self.imgsize - 16, self.objsize):
                for w_start in np.arange(0, self.imgsize, self.objsize):
                    curr_envmap = self.envmaps[i, :, h_start:h_start + self.objsize, w_start:w_start + self.objsize]
                    # print(np.abs(np.mean(curr_envmap.cpu().numpy() - obj.cpu().numpy())))
                    if np.abs(
                            np.mean(curr_envmap.cpu().numpy() - obj.cpu().numpy())) < obj_thres:  # add the configuration to the list
                        # print("add new configuration")
                        for slot_count in range(self.nslots):
                            curr_attnmap = attnmaps[i, slot_count, h_start:h_start + self.objsize,
                                           w_start:w_start + self.objsize]
                            configs.append(curr_attnmap)
        return configs

    def EntropyConfig(self, configs):
        configs = torch.stack(configs)
        data = configs.reshape(configs.shape[0], -1)
        # Calculate the probability density function of the data using a KDE
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(data.cpu().numpy())
        print("after kde?")
        log_density = kde.score_samples(data.cpu().numpy())
        print("after log density?")
        density = np.exp(log_density)
        p = density / np.sum(density)

        # Calculate the entropy of the dataset
        H = entropy(p)
        print('Entropy of the dataset:', H)

        return H

    def ObjectDict(self, envmaps, obj_dict, obj_dict_mask):
        # for given envmaps, the exhausitive list of objects and their representations
        # method: identify the same object with normalization of the object configuration
        # return sequence of object labled frame maps
        dist = []
        # obj_dict = {}
        n_objs = int(self.imgsize / self.objsize)
        assert n_objs == self.imgsize / self.objsize
        n_frames = envmaps.shape[0]
        frame_obj_maps = np.zeros((n_frames, n_objs, n_objs))
        thres = 2
        for i in range(n_frames):
            # print(i)
            for j, h_start in enumerate(np.arange(0, self.imgsize, self.objsize)):
                for k, w_start in enumerate(np.arange(0, self.imgsize, self.objsize)):

                    curr_envmap = envmaps[i, :, h_start:h_start + self.objsize, w_start:w_start + self.objsize].cpu()
                    plt.figure()
                    plt.imshow(curr_envmap.permute(1, 2, 0))
                    plt.savefig(
                        "/home/xuan/projects/def-bashivan/xuan/crafter-ood/crafter_attention/crafter-curated/tile.png")
                    # curr_envmap = (curr_envmap - np.mean(curr_envmap))/(np.std(curr_envmap)+1e-4)
                    # L2 norm normalization => cause Nan value
                    # print("norm of the orignal object:", np.linalg.norm(curr_envmap))
                    # curr_envmap = curr_envmap / (np.linalg.norm(curr_envmap))

                    # print("curr_envmap shape:", curr_envmap.shape)

                    curr_dist = []
                    print(len(obj_dict))

                    for index, (key, value) in enumerate(obj_dict.items()):
                        # dist.append(np.sum(np.abs(curr_envmap - value)))

                        # dist.append(np.linalg.norm(curr_envmap - value))
                        # calcualte distance with L2 norm
                        # print(curr_envmap.shape)

                        diff = curr_envmap.permute(1, 2, 0) - value[:, :, :3]
                        if index == 9:
                            diff9 = diff
                            print("diff9 shape:", diff9.shape)
                            print(diff9[:, :, 0])
                            plt.figure()
                            plt.imshow(diff9)
                            plt.savefig(
                                "/home/xuan/projects/def-bashivan/xuan/crafter-ood/crafter_attention/crafter-curated/diff9.png")

                        if index == 55:
                            diff55 = diff
                            print("diff55 shape:", diff55.shape)
                            print(diff55[:, :, 0])
                            print("diff between diff55 and diff9:", torch.sum(diff9 - diff55))
                            plt.figure()
                            plt.imshow(diff55 - diff9)
                            plt.savefig(
                                "/home/xuan/projects/def-bashivan/xuan/crafter-ood/crafter_attention/crafter-curated/diff55_9.png")

                        mask_diff = (diff[obj_dict_mask[key], :]) / np.sum(obj_dict_mask[key][::])
                        if index == 9 or index == 55:
                            # curr_mask = np.repeat(np.expand_dims(obj_dict_mask[key], axis = 2), 3,axis = 2)
                            plt.figure()
                            plt.imshow(value)
                            plt.savefig(
                                "/home/xuan/projects/def-bashivan/xuan/crafter-ood/crafter_attention/crafter-curated/template_object_%d.png" % index)

                            # plt.figure()
                            # plt.imshow(diff)
                            # plt.savefig("/home/xuan/projects/def-bashivan/xuan/crafter-ood/crafter_attention/crafter-curated/diff%d.png"%index)

                        curr_dist.append(torch.sum(mask_diff[::] ** 2) / np.sum(obj_dict_mask[key][::]))
                        # curr_dist.append(np.linalg.norm(mask_diff))
                        # print(curr_dist)
                        # mask_curr_envmap = curr_envmap[:,obj_dict_mask[key]]
                        # if np.linalg.norm(mask_curr_envmap - value) < thres:
                        # if np.sum(np.abs(curr_envmap - value))< thres:
                        # obj_dict[key] = curr_envmap
                        # is_break = True
                        # frame_obj_maps[i,j,k] = int(key)
                    print(curr_dist)
                    obj_index = np.argmin(curr_dist)
                    frame_obj_maps[i, j, k] = obj_index
                    # np.where(curr_dist == min(curr_dist))
                    print("this is obj %d:" % obj_index)
                    print("mininum dist:", min(curr_dist))
                    print("distance to obj 9:", curr_dist[9])

                    # else: print("difference between two objects:", np.sum(np.abs(curr_envmap - value)))

                    # if index == len(obj_dict)-1 and not is_break:
                    # new_append = True
                    # print("new append at location:", i,j,k)

                    # if new_append:
                    # obj_dict["%d" % (len(obj_dict) + 1)] = curr_envmap

        # plt.figure()
        # plt.hist(dist)
        # plt.savefig("/home/xuan/projects/def-bashivan/xuan/crafter-ood/crafter_attention/results/envmaps/t_50000_p_1_det_thres_0.01/dist.png")

        return obj_dict, frame_obj_maps

    def NonObject_Transformations(self, envmaps):
        """
            Input:
                sequence of environment maps
            Output:
                sequence of transformation matrix
        """
        trans_matrix = []
        for i in range(envmaps.shape[0] - 1):
            trans_matrix.append(envmaps[i + 1, :, :, :] / envmaps[i, :, :, :])
        return trans_matrix

    def plotAttnDist(self, attnmaps, savepath=None):
        plotdata = attnmaps.cpu().numpy()

        plt.figure()

        # Create subplots
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        fig.subplots_adjust(hspace=0.4)  # Adjust vertical spacing between subplots

        fig.suptitle('Histogram of Attention Intensity', fontsize=14, fontweight='bold')

        # Iterate over each subplot
        for i, ax in enumerate(axes.flat):
            curr_data = plotdata[:, i, :, :].reshape(-1)
            ax.hist(curr_data, bins=30, alpha=0.6, color="b")
            ax.set_yscale('log', nonposy='clip')
            ax.set_title(f'slot {i + 1}')
            ax.set_xlabel('Attention Intensity')
            if i == 0 or i == 4:
                ax.set_ylabel('Frequency')

            if curr_data.max() > 1:
                inset_data = curr_data[curr_data > 1]
                # Create inset figure
                inset_ax = ax.inset_axes([0.6, 0.6, 0.25, 0.25])  # [left, bottom, width, height]
                inset_ax.hist(inset_data, bins=30, alpha=0.6, color="b")
                inset_ax.set_xticks([])
                inset_ax.set_yticks([])

        plt.savefig(savepath)

    def binaryMap(self, attnmaps, thres=1e-1):
        """return binary version of the attnmap cutoff by the given threshold"""
        return (attnmaps > thres).to(torch.int)


def obj_template_generation(pathtofile):
    obj_dict = {}
    for i, filename in enumerate(os.listdir(pathtofile)):
        file_path = os.path.join(pathtofile, filename)
        image = imageio.imread(file_path)
        obj_dict["%d" % i] = image
        # print(image[:,:, -1])

        # if i == 55:
        #     plt.figure()
        #     plt.imshow(image)
        #     plt.savefig("/home/xuan/projects/def-bashivan/xuan/crafter-ood/crafter_attention/crafter-curated/object_dict/object%d.png" % i)
    return obj_dict


def obj_dict_mask_gen(obj_dict):
    obj_dict_mask = {}
    for i, (key, obj) in enumerate(obj_dict.items()):
        if obj.shape[-1] == 3:
            obj_dict_mask["%d" % i] = (np.ones(obj.shape[:2]) == 1)
        else:
            obj_dict_mask["%d" % i] = (obj[:, :, -1] != 0)
        # print(obj_dict_mask["%d" % i].shape)
    return obj_dict_mask


obj_dict = obj_template_generation("/home/xuan/projects/def-bashivan/xuan/crafter-ood/crafter_attention/assets")
obj_dict_mask = obj_dict_mask_gen(obj_dict)
# print(obj_dict_mask)

#### plot attention instensity distribution over training episodes for models with various patch size
#### why: looking for attention intensity changing patterns over time and across time slots and patchsizes
#### why: to determine the cutoff threshold for binary attention configuration
# phases = ["det"]
# patch_sizes = [1,2,4,8,12,16,20]
# training_eps = [50000,150000,350000,850000,2050000,4050000]
# # patch_sizes = [1,2]
# # training_eps = [50000, 150000,350000]

# n_slots = 8

# fig_root = "/home/xuan/projects/def-bashivan/xuan/crafter-ood/crafter_attention"
# root = "/home/xuan/projects/def-bashivan/xuan/crafter-ood/crafter_attention/crafter-curated"
# max_attn = np.zeros((len(patch_sizes), len(training_eps), len(phases), 8)) # 8 is the number of slots
# for pi, patch_size in enumerate(patch_sizes):
#     for ti, training_ep in enumerate(training_eps):
#         for ipa, phase in enumerate(phases):
#             print("patchsize %d, training episodes %d, phase %s" % (patch_size, training_ep, phase))
#             path2slotmasks = os.path.join(root, "t_%d"%training_ep, "patch_size_%d_valid_%s_slot_masks" % (patch_size, phase))
#             path2attnmaps = os.path.join(root, "t_%d"%training_ep, "patch_size_%d_valid_%s_attn_maps" % (patch_size, phase))
#             path2envinfo = os.path.join(root, "t_%d"%training_ep, "patch_size_%d_valid_%s_episode_details" % (patch_size, phase))
#             path2envmaps = os.path.join(root, "t_%d"%training_ep, "patch_size_%d_valid_%s_episode_observations" % (patch_size, phase)) # missing env maps

#             metric_cal = AttnMetric(objsize=8, path2attnmaps = path2attnmaps, path2slotmasks = path2slotmasks,
#                         path2envmaps = path2envmaps, path2envinfo = path2envinfo)

#             plotdata = metric_cal.slotmasks.cpu().numpy()
#             for slot in range(plotdata.shape[1]): # number of slots
#                 max_attn[pi, ti, ipa, slot] = max(plotdata[:,slot,:,:].reshape(-1))

# print(metric_cal.slotmasks.shape)


# fig_savepath = os.path.join(fig_root, "figures", "attn_dist", "patchsize_%d_eps_%d_%s.jpeg" % (patch_size, training_ep, phase))
# metric_cal.plotAttnDist(metric_cal.slotmasks, savepath = fig_savepath)

# for slot in range(8):
#     plt.figure()
#     for ti in range(len(training_eps)):
#         plt.plot(max_attn[:,ti, 0, slot], label = "eps = %d" % training_eps[ti])
#     plt.legend()
#     plt.xlabel("patchsize")
#     plt.ylabel("max attention")
#     fig_savepath = os.path.join(fig_root, "figures", "attn_dist", "slot_%d.jpeg" % (slot))
#     plt.savefig(fig_savepath)


#################################################################################
# count total number of attention configurations

# thres = 10
# phases = ["det", "sto"]
# patch_sizes = [1,2,4,8,12,16,20]
# training_eps = [50000,150000,350000,850000,2050000,]


# n_slots = 8

# fig_root = "/home/xuan/projects/def-bashivan/xuan/crafter-ood/crafter_attention"
# root = "/home/xuan/projects/def-bashivan/xuan/crafter-ood/crafter_attention/crafter-curated"
# n_configs_dict = {}
# for patch_size in patch_sizes:
#     for training_ep in training_eps:
#         for phase in phases:
#             print("patchsize %d, training episodes %d, phase %s" % (patch_size, training_ep, phase))
#             path2slotmasks = os.path.join(root, "t_%d"%training_ep, "patch_size_%d_valid_%s_slot_masks" % (patch_size, phase))
#             path2attnmaps = os.path.join(root, "t_%d"%training_ep, "patch_size_%d_valid_%s_attn_maps" % (patch_size, phase))
#             path2envinfo = os.path.join(root, "t_%d"%training_ep, "patch_size_%d_valid_%s_episode_details" % (patch_size, phase))
#             path2envmaps = os.path.join(root, "t_%d"%training_ep, "patch_size_%d_valid_%s_episode_details" % (patch_size, phase)) # missing env maps

#             metric_cal = AttnMetric(objsize=8, path2attnmaps = path2attnmaps, path2slotmasks = path2slotmasks,
#                         path2envmaps = path2envmaps, path2envinfo = path2envinfo)

#             _, l = metric_cal.countAttnConfigBinary(attnmaps = metric_cal.slotmasks[:20], thres = thres)
#             n_configs_dict["ps%d_te%d_p%s" % (patch_size, training_ep, phase)] = l
#             print("\t%d possible configurations " % (l,))
# print(n_configs_dict)

# # Save the dictionary
# with open('/home/xuan/projects/def-bashivan/xuan/crafter-ood/crafter_attention/results/total_configs_binary_thres%.2f.json' % thres, 'w') as file:
#     json.dump(n_configs_dict, file)

# # visualization

# phases = ["det"]
# patch_sizes = [1,2,4,8,12,16,20]
# training_eps = [50000,150000,350000,850000,2050000,]
# thress = [0.01, 0.10, 1.00, 10.00]
# for thres in thress:
#     with open('/home/xuan/projects/def-bashivan/xuan/crafter-ood/crafter_attention/results/total_configs_binary_thres%.2f.json' % thres, 'r') as file:
#         n_configs_dict = json.load(file)
#     save_root = "/home/xuan/projects/def-bashivan/xuan/crafter-ood/crafter_attention/results/total_binary_configs"
#     phase = phases[0]
#     plt.figure()

#     for patch_size in patch_sizes:
#         data = []
#         for training_ep in training_eps:
#             data.append(n_configs_dict["ps%d_te%d_p%s" % (patch_size, training_ep, phase)])

#         plt.plot(training_eps, data ,label = "patch size %d" % patch_size)

#     plt.xscale('log',)
#     plt.ylabel("# binary configurations")
#     plt.xlabel("training episodes")
#     plt.legend()
#     # Adjust the spacing between subplots
#     plt.savefig(os.path.join(save_root, "n_confgis_vs_trainingeps_thres_%.2f.jpg" % thres))

# # for the final training_eps, plot the changes of n_configs along patch_size
# training_ep = training_eps[-1]
# data = []
# for patch_size in patch_sizes:
#     data.append(n_configs_dict["ps%d_te%d_p%s" % (patch_size, training_ep, phase)])
# plt.figure()
# plt.plot(patch_sizes,data)
# plt.xscale('log',)
# plt.xlabel("patch_size")
# plt.ylabel("# binary configurations")
# plt.title("number of binary attention configurations at the final training episode")
# plt.savefig(os.path.join(save_root, "n_configs_vs_patch_size_thres_%.2f.jpg" % thres))


##################################################################################
### goal: to identify representation of one unique object configurations
### current status: still with environment maps from Chris's data
### current statue: only focus on the no-object frame to compare the changes across frames
### DONE!

####### for each file, save the object reference and frame frame_obj_maps
# phases = ["det", "sto"]
# patch_sizes = [1,2,4,8,12,16,20]
# training_eps = [50000,150000,350000,850000,2050000,4050000]
phases = ["det"]
patch_sizes = [1]
training_eps = [50000]
thres = 0.01
obj_size = 16

n_slots = 8

fig_root = "/home/xuan/projects/def-bashivan/xuan/crafter-ood/crafter_attention"
root = "/home/xuan/projects/def-bashivan/xuan/crafter-ood/crafter_attention/crafter-curated"
save_root = "/home/xuan/projects/def-bashivan/xuan/crafter-ood/crafter_attention/results/envmaps"
for patch_size in patch_sizes:
    for training_ep in training_eps:
        for phase in phases:
            print("patchsize %d, training episodes %d, phase %s" % (patch_size, training_ep, phase))
            curr_save_dir = os.path.join(save_root, "t_%d_p_%d_%s_thres_%.2f" % (training_ep, patch_size, phase, thres))
            if not os.path.isdir(curr_save_dir):
                os.mkdir(curr_save_dir)
            path2slotmasks = os.path.join(root, "t_%d" % training_ep,
                                          "patch_size_%d_valid_%s_slot_masks" % (patch_size, phase))
            path2attnmaps = os.path.join(root, "t_%d" % training_ep,
                                         "patch_size_%d_valid_%s_attn_maps" % (patch_size, phase))
            path2envinfo = os.path.join(root, "t_%d" % training_ep,
                                        "patch_size_%d_valid_%s_episode_details" % (patch_size, phase))
            path2envmaps = os.path.join(root, "t_%d" % training_ep, "patch_size_%d_valid_%s_episode_observations" % (
            patch_size, phase))  # missing env maps

            metric_cal = AttnMetric(objsize=obj_size, path2attnmaps=path2attnmaps, path2slotmasks=path2slotmasks,
                                    path2envmaps=path2envmaps, path2envinfo=path2envinfo)

            # #             print("start of each eps:", metric_cal.episodes_start)

            #             # for sanity check of the generated frame_obj_maps
            # for ti in [0]:
            #     plt.figure()
            #     plt.imshow(np.transpose(metric_cal.envmaps[ti].cpu().numpy(),(1,2,0)))
            #     print(curr_save_dir + "/envmap_frame_%d.jpeg" % ti)
            #     plt.savefig(curr_save_dir + "/envmap_frame_%d.jpeg" % ti)

            obj_dict, frame_obj_maps = metric_cal.ObjectDict(metric_cal.envmaps[:1, :, :16, 16:32], obj_dict,
                                                             obj_dict_mask)
            plt.figure()
            plt.imshow(metric_cal.envmaps[0, :, :, :].permute(1, 2, 0))
            plt.savefig(
                "/home/xuan/projects/def-bashivan/xuan/crafter-ood/crafter_attention/crafter-curated/env_frame1.jpeg")
            # for key in obj_dict.keys():
            #     plt.figure()
            #     print("current save object image shape:", np.transpose(obj_dict[key],(1,2,0)).shape)
            #     plt.imshow(np.transpose(obj_dict[key],(0,1,2)))
            #     plt.savefig(os.path.join(curr_save_dir,"object_%s.png" % key))
            print(frame_obj_maps[0])
#             with open(os.path.join(curr_save_dir, "obj_dict.pkl"), "wb") as file:
#                 pickle.dump(obj_dict, file)
#             json_frame_obj_maps = frame_obj_maps.tolist()
#             with open(os.path.join(curr_save_dir,"frame_obj_maps.json"), 'w') as file:
#                 json.dump(json_frame_obj_maps, file)

#             # print(frame_obj_maps)
#             print(obj_dict.keys())
# n_configs_per_obj = []
# for ti in range(1,len(obj_dict.keys())+1):
#     possible_configs = metric_cal.countAttnConfigBinaryperObject(metric_cal.slotmasks[:100], ti, frame_obj_maps[:100], thres = thres)
#     print("object %d, number of attention configs:" %ti, len(possible_configs))
#     n_configs_per_obj.append(len(possible_configs))

# plt.figure()

# plt.bar(np.arange(len(n_configs_per_obj)),n_configs_per_obj,width = 0.4)
# plt.xlabel("object")
# plt.ylabel("# configurations")
# plt.savefig("/home/xuan/projects/def-bashivan/xuan/crafter-ood/crafter_attention/results/binary_configs_per_object/patchsize%d_training%d_phase%s_thres%.2f_nconfigs_perobject_partial.jpg" % (patch_size, training_ep, phase, thres))

# for i in range(9):
#     t1 = trans_matrix[i][:,:8,:8].cpu().numpy()
#     t2 = (metric_cal.envmaps[i+1,:, 48:56, 8:16].cpu().numpy()/metric_cal.envmaps[i,:, 56:64, 16:24].cpu().numpy())
#     print(np.sum(t1.reshape(-1) == t2.reshape(-1))/len(t1.reshape(-1)))
# try normalization and match!!!!

# norm1 = metric_cal.envmaps[0,:,48:56, 8:16].permute(1,2,0).cpu().numpy()
# norm1 = (norm1 - np.mean(norm1))/np.std(norm1)
# plt.figure()
# plt.imshow(norm1)
# plt.savefig("/home/xuan/projects/def-bashivan/xuan/crafter-ood/crafter_attention/figures/env_map/normalized_1.png")
# norm2 = metric_cal.envmaps[1,:,8:16, 8:16].permute(1,2,0).cpu().numpy()
# norm2 = (norm2 - np.mean(norm2))/np.std(norm2)
# plt.figure()
# plt.imshow(norm2)
# plt.savefig("/home/xuan/projects/def-bashivan/xuan/crafter-ood/crafter_attention/figures/env_map/normalized_2.png")
# print(np.sum(np.abs(norm1 - norm2)))


# print(np.sum(trans_matrix[1][:8,:8].cpu().numpy() == metric_cal.envmaps[2,:, 16:24, 8:16].cpu().numpy()/metric_cal.envmaps[1,:, 16:24, 8:16].cpu().numpy()))


# below is not correct: should not not comparing consecutive frames with new or diaappeared objects
# for i in range(10-1):
#     standard = trans_matrix[i][:,:8,:8].cpu().numpy()
#     for j in range(9):
#         for k in range(9):
#             target = metric_cal.envmaps[i+1,:,j*8:(j+1)*8,k*8:(k+1)*8].cpu().numpy()/ metric_cal.envmaps[i,:,j*8:(j+1)*8,k*8:(k+1)*8].cpu().numpy()
#             if np.sum(np.abs(target - standard)) > 1e-2:
#                 print("frame %d location %d %d:" % (i,j,k))
#                 print(np.sum(np.abs(target - standard)))
# I need to calculate the transition matrix by object type; can not do it across object
# for i in range(len(trans_matrix)):
#     for j in range(9):
#         for k in range(9):
#             if np.sum(np.abs(trans_matrix[i][:,8*j:8*(j+1),8*k:8*(k+1)].cpu().numpy() - trans_matrix[i][:,:8, :8].cpu().numpy())) > 1:
#                 print("frame %d, location %d, %d" % (i,j,k))


###  visualization of the environment map
# for i in range(10):
#     plt.figure()
#     plt.imshow(metric_cal.envmaps[i,:,:,:].permute(1,2,0).cpu().numpy())
#     plt.savefig("/home/xuan/projects/def-bashivan/xuan/crafter-ood/crafter_attention/figures/" +  "env_map/" + "%d.png" % i)

# check object pixel level representation for non-object tiles: is multiplication relationshi?
# take the upperleft cornor non-object tile and plot the distribution
# fig, axes = plt.subplots(2, 5, figsize=(12, 6))
# fig.subplots_adjust(hspace=0.4)  # Adjust vertical spacing between subplots

# # Iterate over each subplot
# colors = ["R","G","B"]
# for j, color in enumerate(colors):
#     for i, ax in enumerate(axes.flat):
#         ax.hist(metric_cal.envmaps[i,j,:,:].cpu().numpy().reshape(-1), bins=30)
#         ax.set_title(f'Figure {i+1}')
#     # Show the plot
#     plt.savefig("/home/xuan/projects/def-bashivan/xuan/crafter-ood/crafter_attention/figures/env_map_dist/non_object_tile_%s.jpeg" % color)

### for non-object tile, perform element-wise division; then compare if the division can be performed on object tiles of the same pairs of frames
# for i in range(10):
#     non_obj_div = metric_cal.envmaps[i+1,:,0:8,0:8] / metric_cal.envmaps[i,:,0:8,0:8] # pay attention to the order!
#     new_tree_rep = metric_cal.envmaps[i,:,8:16, 40:48] * non_obj_div
#     # check if there is a difference between calcualted and real tree representation
#     print(np.sum(np.abs(new_tree_rep.cpu().numpy() - metric_cal.envmaps[i+1, :,8:16,40:48].cpu().numpy())))
#     # obj_div = metric_cal.envmaps[i,:,8:16, 40:48] / metric_cal.envmaps[i+1,:,8:16, 40:48]

# a working approach to tag each frame with sequence of object labels
# step 1: for each frame: identify representation of non-object representation
#### how: save template normalized non-object tile, the compare with tiles from the given frame with normalization
# step 2: tag each frame with object labels

# to save: dict obj_rep = {"object name:", most up to date representations}
# to save: frames_objs_map = (n_frames, n_objs, n_objs) (index of each object)
# to save: transformation matrix between two consecutive frames (n_frames - 1, 3, img_size, img_size)

#### iterate over frames [at frame i]:
######## identify transformation matrix between frame i and i-1
######## generate updated object representation for current frame
######## compare and update current object representation by loop through the current frame


### a potential approach: identify the transformation matrix => not working!!!!!
# def id_array_transformation(array1, array2):
#     # Calculate the transformation matrix
#     transformation_matrix = np.linalg.inv(array1) @ array2
#     return transformation_matrix

# transformation_matrixs = []
# for i in range(3):
#     ## lead to error: numpy.linalg.LinAlgError: Singular matrix
#     transformation_matrix = id_array_transformation(metric_cal.envmaps[i, :, 0:8, 0:8].cpu().numpy(), metric_cal.envmaps[i+1, :, 0:8, 0:8].cpu().numpy())
#     transformation_matrixs.append(transformation_matrix)
#     print(np.array_equal(transformation_matrixs[-2],transformation_matrixs[-1]))

# print(np.array_equal(metric_cal.envmaps[0, 0, 0:8, 0:8].cpu().numpy(), metric_cal.envmaps[1, 0, 0:8,0:8].cpu().numpy()))

# check if other object tile follows the same representation change as the non-object tiles


####### count total number of possible configurations
# path2attnmaps = "./debug_attn_patch_8_stride_8/kind-sky-14/valid_det/attn_maps_5"
# path2slotmasks = "./debug_attn_patch_8_stride_8/kind-sky-14/valid_det/slot_masks_5"
# path2envmaps = "./debug_attn_patch_8_stride_8/kind-sky-14/valid_det/episode_observations_5"
# path2envinfo = "./debug_attn_patch_8_stride_8/kind-sky-14/valid_det/episode_details_5"


# metric_cal = AttnMetric(objsize=8, path2attnmaps = path2attnmaps, path2slotmasks = path2slotmasks,
#                         path2envmaps = path2envmaps, path2envinfo = path2envinfo)

# # count the total number of configurations for given episode number
# thres = 0.1
# for eps_count in range(len(metric_cal.episodes_start)):
#     _, l = metric_cal.countAttnConfigBinary(attnmaps = metric_cal.slotmasks[metric_cal.episodes_start[eps_count]:metric_cal.episodes_start[eps_count+1],:,:], thres = thres)
#     print("%d possible configurations for episode %d" % (l, eps_count))
#     break

# possible configurations per object, for given episode
# for eps_count in range(len(metric_cal.episodes_start)):
# configs, obj_list = metric_cal.countAttnConfigBinaryperObject(attnmaps = metric_cal.slotmasks[metric_cal.episodes_start[eps_count]:metric_cal.episodes_start[eps_count+1],:,:], thres = 0.1)
# print("possible number of object:", obj_list)
# print("possible configurations per object:", configs)
# break

# generate list of continuous configurations of attention for a give object within a given episode
# obj_list, l = metric_cal.ObjectDict()
# print("number of objects:", l)
# for eps_count in range(len(metric_cal.episodes_start)):
#     configs = metric_cal.ContinuousAttentionperObject(metric_cal.slotmasks[metric_cal.episodes_start[eps_count]:metric_cal.episodes_start[eps_count+1],:,:], obj_list[2], obj_thres = 1e-2)
#     break
# print(len(configs))

# entropy = metric_cal.EntropyConfig(configs)
# print("entropy for object 0:", entropy)


#  visualization of the environment map
# for i in range(20,40):
#     plt.figure()
#     plt.imshow(metric_cal.envmaps[i,:,:,:].permute(1,2,0).cpu().numpy())
#     plt.savefig("/home/xuan/projects/def-bashivan/xuan/crafter-ood/crafter_attention/figures/" + "%d.png" % i)


# check object pixel level representation
# print(metric_cal.envmaps[0, 0, 0:8, 0:8].cpu().numpy())
# print(metric_cal.envmaps[0, 1, 0:8, 0:8].cpu().numpy())
# print(np.array_equal(metric_cal.envmaps[0, 0, 0:8, 0:8].cpu().numpy(), metric_cal.envmaps[1, 0, 0:8,0:8].cpu().numpy()))

# means = []
# for i in range(4):
#     for j in range(i+1,4):
#         means.append(np.mean(metric_cal.envmaps[0, :, 56:64, i*8:(i+1)*8].cpu().numpy() -metric_cal.envmaps[0, :, 56:64, j*8:(j+1)*8].cpu().numpy()))
#         print(i,j, means[-1])

# plt.figure()
# plt.hist(means)
# plt.savefig("/home/xuan/projects/def-bashivan/xuan/crafter-ood/crafter_attention/figures/" + "means_dist_stock_vs_otherstock.png")


# plt.figure()
# plt.imshow(metric_cal.envmaps[i+1,:,:8,:8].permute(1,2,0).cpu().numpy() - metric_cal.envmaps[i,:,:8,:8].permute(1,2,0).cpu().numpy())
# plt.savefig("/home/xuan/projects/def-bashivan/xuan/crafter-ood/crafter_attention/figures/" + "tile0_img_diff_%d.png" % i)

# print(np.array_equal(metric_cal.envmaps[0, :, 8*1:8*2,8*5:8*6].cpu().numpy(), metric_cal.envmaps[0, :, 8*1:8*2,8*6:8*7].cpu().numpy()))


# thres = 0.1
# plot the distribution of attention intensity
# metric_cal.plotAttnDist(metric_cal.slotmasks, savepath = "/home/xuan/projects/def-bashivan/xuan/crafter-ood/crafter_attention/figures/debug_attndist.jpeg")

# count the number of possible object and their corresponding representations
# obj_list, l = metric_cal.ObjectDict()
# print("number of objects:", l)
# # for i in range(4):
# #     print(np.abs(np.mean(obj_list[3].cpu().numpy() - metric_cal.envmaps[0, :, 56:64, i*8:(i+1)*8].cpu().numpy())))
# for i, obj in enumerate(obj_list):
#     plt.figure()

#     plt.imshow(obj.permute(1,2,0).cpu().numpy())
#     plt.savefig("/home/xuan/projects/def-bashivan/xuan/crafter-ood/crafter_attention/figures/" + "tile_%d.png" % i)
# print(np.array_equal(obj_list[0].permute(1,2,0).cpu().numpy(),obj_list[1].permute(1,2,0).cpu().numpy()))


# print(metric_cal.attnmaps.shape)
# print(metric_cal.attnmaps.max())
# print(metric_cal.attnmaps.min())

# print(metric_cal.slotmasks.shape)
# print(metric_cal.envmaps.shape)
# print(metric_cal.envinfo)

