import numpy as np
import torch
from lib.models.blocks.lora_prune import LPConv, ABConv
from loralib import ConvLoRA


def _is_target_larer(module):
    return (isinstance(module, LPConv) or isinstance(module, ABConv)) and module.is_prune
    # return (isinstance(module, ConvLoRA))

def init_sensitivity_dict(model):
    sensitivity_record = {}
    for name, module in model.named_modules():
        if _is_target_larer(module):
            groups = module.conv.weight.shape
            # print(groups)
            # name = ".".join(name.split('.')[:-1])
            # print(name)
            if name in sensitivity_record:
                continue
            sensitivity_record[name] = module.lora_A.weight.new_zeros(groups)
    return sensitivity_record

def update_sensitivity_dict(model, s_dict, pruning_type='lora'):
    s_all = init_sensitivity_dict(model)
    for name, module in model.named_modules():
        if _is_target_larer(module):
            # is_attn = name.split('.')[-1] in pruning_groups['self_attn']
            # fan_in = name.split('.')[-1] in pruning_groups['block']
            s = compute_sensitivity(module, pruning_type)
            # print(s.shape)
            # print(name)
            # name = ".".join(name.split('.')[:-1])
            s_all[name] = s
            # print(name)
            # print(s_all[name].shape)
            #s_dict[i] += s
    for name, imp in s_all.items():
        if torch.isnan(imp.sum()):
            return s_dict
    for name, imp in s_dict.items():
        # print(name)
        # print(imp.shape)
        s_dict[name] = imp * 0.9 + s_all[name] * 0.1
    return s_dict

def compute_sensitivity(layer, prune_metric='lora', transpose=False, norm=True):
    a = layer.lora_A.weight.data
    b = layer.lora_B.weight.data
    if prune_metric == 'lora':
        grad_a = layer.lora_A.grad
        grad_b = layer.lora_B.grad
        grad = 1#(grad_b @ a + b @ grad_a - grad_b @ grad_a)
    elif prune_metric == 'magnitude':
        grad = 1
    elif prune_metric == 'grad':
        grad = layer.conv.weight.grad
    else:
        raise NotImplementedError
    if hasattr(layer, 'state'):
        weight = (layer.conv.weight.data * layer.state.SCB.reshape(-1, 1)) / 127
    else:
        weight = layer.conv.weight.data
    s = (grad * ((b @ a).view(layer.conv.weight.shape) * layer.scaling + weight)).abs()
    if transpose:
        s = s.t()
    # s = s.sum(1)
    # if norm:
    #     s = s / (torch.linalg.norm(s) + 1e-8)
    return s

def global_prune(model, s_dict, ratio):
    parameters = []
    masks = []
    sensitivity = []

    for name, module in model.named_modules():
        if _is_target_larer(module):
            parameters.append(module.conv.weight)
            # mask_0 = torch.ones_like(module.lora_mask.data)
            masks.append(torch.ones_like(module.lora_mask.data))
            # importance = s_dict[name] * module.lora_mask.data
            sensitivity.append(s_dict[name] * module.lora_mask.data)
    # flatten importance scores to consider them all at once in global pruning
    all_param = torch.nn.utils.parameters_to_vector(parameters)
    # similarly, flatten the masks (if they exist), or use a flattened vector
    # of 1s of the same dimensions as t
    final_mask = torch.nn.utils.parameters_to_vector(masks)
    relevant_importance_scores = torch.nn.utils.parameters_to_vector(sensitivity)
    # print(relevant_importance_scores)

    # use the `compute_mask` method from `PruningContainer` to combine the
    # mask computed by the new method with the pre-existing mask
    # total_num = all_param.nelement()
    need_prune_num = int(all_param.nelement() * ratio)
    # argsort return idx
    # can_prune = torch.argsort(relevant_importance_scores)[:need_prune_num]
    # print(can_prune)
    final_mask[torch.argsort(relevant_importance_scores)[:need_prune_num]] = 0

    del all_param
    del relevant_importance_scores

    # original_param_num = 0
    # pruned_param_num = 0
    pointer = 0
    for name, module in model.named_modules():
        if _is_target_larer(module):
            # original_param_num += np.prod(module.conv.weight.shape)
            # pruned_param_num += np.prod(module.conv.weight.shape) * ratio

            param = getattr(module.conv, 'weight')
            # The length of the parameter
            num_param = module.conv.weight.numel()
            # Slice the mask, reshape it
            module.lora_mask.data = final_mask[pointer : pointer + num_param].view_as(param)
            # Increment the pointer to continue slicing the final_mask
            pointer += num_param
    del final_mask
    # print("pruned/original parameters number:{:3f}/{:3f}  ratio:{:3f}".format(pruned_param_num*1e-6,
                                                                            #    original_param_num*1e-6,
                                                                            #    pruned_param_num/original_param_num))

def local_prune(model, s_dict, ratio, target_ratio):
    original_param_num = 0
    pruned_param_num = 0
    for name, module in model.named_modules():
        if _is_target_larer(module):
            original_param_num += np.prod(module.conv.weight.shape)
            pruned_param_num += np.prod(module.conv.weight.shape) * ratio
            # is_attn = name.split('.')[-1] in pruning_groups['self_attn']
            # if name.split('.')[-1] in pruning_groups['block']:
            #     continue
            # name = ".".join(name.split('.')[:-1])
            if not hasattr(module, 'lora_mask'):
                continue
            if (1-module.lora_mask.mean()).item() >= target_ratio:
                continue
            total_num = module.lora_mask.numel()
            c_mask = module.lora_mask.data
            mask = torch.ones_like(c_mask)

            # if is_attn:
            #     mask = mask.reshape(-1, DIM)[:, 0]
            #     c_mask = c_mask.reshape(-1, DIM)[:, 0]
            #     total_num /= DIM
            need_prune_num = int(total_num * ratio)
            importance = s_dict[name] * c_mask
            can_prune = torch.argsort(importance)[:need_prune_num]
            mask[can_prune] = 0
            # if is_attn:
            #     mask = (mask.new_ones(module.lora_mask.shape).reshape(-1, DIM) * mask.unsqueeze(1)).reshape(-1)
            module.lora_mask.data = mask
        else:
            if hasattr(module, 'weight'):
                original_param_num += np.prod(module.weight.shape)
    print("pruned/original parameters number:{:3f}/{:3f}  ratio:{:3f}".format(pruned_param_num*1e-9,
                                                                               original_param_num*1e-9,
                                                                               pruned_param_num/original_param_num))

def schedule_sparsity_ratio(
    step,
    total_step,
    initial_warmup,
    final_warmup,
    initial_sparsity,
    final_sparsity,
):
    if step <= initial_warmup * total_step:
        sparsity = initial_sparsity
    elif step > (total_step - final_warmup * total_step):
        sparsity = final_sparsity
    else:
        spars_warmup_steps = initial_warmup * total_step
        spars_schedu_steps = (final_warmup + initial_warmup) * total_step
        mul_coeff = 1 - (step - spars_warmup_steps) / (total_step - spars_schedu_steps)
        sparsity = final_sparsity + (initial_sparsity - final_sparsity) * (mul_coeff ** 3)
    return sparsity
