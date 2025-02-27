import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from loralib import LoRALayer



def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    if bias =='depth':
        for n, m in model.named_modules():
            if isinstance(m, ABConv):
                m.conv.weight.requires_grad = False
    else:   
        for n, p in model.named_parameters():
            # print(n)
            if ("lora_" not in n) and ("31.m" not in n):
                p.requires_grad = False
            # if "lora_" not in n:
            #     p.requires_grad = False
        if bias == 'none':
            return
        elif bias == 'all':
            for n, p in model.named_parameters():
                if 'bias' in n:
                    p.requires_grad = True
        elif bias == 'bn':
            for n, p in model.named_parameters():
                if 'bn' in n:
                    p.requires_grad = True
        elif bias == 'lora_only':
            for m in model.modules():
                if isinstance(m, LoRALayer) and \
                    hasattr(m, 'bias') and \
                    m.bias is not None:
                        m.bias.requires_grad = True
        else:
            raise NotImplementedError
    
    
# def dora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
#     my_state_dict = model.state_dict()
#     if bias == 'none':
#         return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'magnitude' in k}
#     elif bias == 'all':
#         return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k or 'magnitude' in k}
#     elif bias == 'bn':
#         return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bn' in k or 'magnitude' in k}
#     elif bias == 'lora_only':
#         to_return = {}
#         for k in my_state_dict:
#             if 'lora_' in k:
#                 to_return[k] = my_state_dict[k]
#                 bias_name = k.split('lora_')[0]+'bias'
#                 if bias_name in my_state_dict:
#                     to_return[bias_name] = my_state_dict[bias_name]
#         return to_return
#     else:
#         raise NotImplementedError
    

### LoRAPrune Convolution layer
class LPConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs):
        super(LPConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)

        self.kernel = kernel_size
        self.lora_mask = nn.Parameter(torch.ones(self.conv.weight.shape), requires_grad=False)

        if r > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
              self.conv.weight.new_zeros((out_channels//self.conv.groups*kernel_size, r*kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.conv.weight.requires_grad = False
            self.is_prune = True
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        super(LPConv, self).train(mode)
        # print(self.conv.weight.shape)
        if mode: # Train
            # if self.merge_weights and self.merged:
            #     if self.r > 0:
            #         # Make sure that the weights are not merged
            #         self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
            self.merged = False
        else: # Test merge weights
            self.conv.weight.data = (self.conv.weight.data + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling) * self.lora_mask
            # if self.merge_weights and not self.merged:
            #     if self.r > 0:
            #         # Merge the weights and mark it
            #         self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
            self.merged = True
        # print(self.merged)

    def forward(self, x):
        if self.r > 0 and not self.merged: # Train
            # print('Train')
            if hasattr(self, 'lora_mask'):
                return self.conv._conv_forward(
                    x, 
                    (self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling) * self.lora_mask,
                    self.conv.bias
                )
            # else:             
            #     return self.conv._conv_forward(
            #         x, 
            #         self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
            #         self.conv.bias
            #     )
        else: # Test
            # print(Test)
            if hasattr(self, 'lora_mask'):
            #     return self.conv._conv_forward(
            #         x, 
            #         self.conv.weight * self.lora_mask,
            #         self.conv.bias
            #     )
            # else:
                # print(self.conv.weight.data)
                return self.conv(x)
            
    def remove(self):
        if self.r > 0 and not self.merged:
            self.conv.weight.data = (self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling) * self.lora_mask
        else:
            self.conv.weight.data = self.conv.weight * self.lora_mask

# ### AB Convolution layer: Depthwise Separable Convolution
class ABConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs):
        super(ABConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        # print(f"O {self.conv.weight.shape}")
        # print(self.conv.weight.shape)
        # self.in_channels = in_channels
        # self.out_channels = out_channels
        # self.kernel_size = kernel_size
        # self.rank = r

        # Initialize low rank metric A and B
        if 'padding' not in kwargs and 'stride' not in kwargs: # Default
            self.lora_A = nn.Conv2d(in_channels, in_channels, kernel_size, groups=in_channels, bias=False)
            # self.lora_A = nn.Conv2d(in_channels, r, kernel_size, groups=in_channels, bias=False)
            # # Depthwise Convolution
            # self.lora_A = nn.Conv2d(in_channels, in_channels, kernel_size, groups=in_channels)
        elif 'padding' in kwargs and 'stride' not in kwargs:
            if kwargs['padding'] == 0:
                self.lora_A = nn.Conv2d(in_channels, in_channels, kernel_size, groups=in_channels, bias=False)
                # self.lora_A = nn.Conv2d(in_channels, r, kernel_size, groups=in_channels, bias=False)
            else:
                self.lora_A = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kwargs['padding'], groups=in_channels, bias=False)
                # self.lora_A = nn.Conv2d(in_channels, r, kernel_size, padding=kwargs['padding'], groups=in_channels, bias=False)
        elif 'padding' not in kwargs and 'stride' in kwargs:
            self.lora_A = nn.Conv2d(in_channels, in_channels, kernel_size, stride=kwargs['stride'], groups=in_channels, bias=False)
            # self.lora_A = nn.Conv2d(in_channels, r, kernel_size, stride=kwargs['stride'], groups=in_channels, bias=False)
        else:
            self.lora_A = nn.Conv2d(in_channels, in_channels, kernel_size, stride=kwargs['stride'], padding=kwargs['padding'], groups=in_channels, bias=False)
            # self.lora_A = nn.Conv2d(in_channels, r, kernel_size, stride=kwargs['stride'], padding=kwargs['padding'], groups=in_channels, bias=False)
        # if 'padding' not in kwargs : # Default
        #     self.padding = 0
        #     # # Depthwise Convolution
        #     # self.lora_A = nn.Conv2d(in_channels, in_channels, kernel_size, groups=in_channels)
        # else:
        #     self.padding = kwargs['padding']

        # if 'stride' not in kwargs:
        #     self.stride = 1 # Default
        # else:
        #     self.stride = kwargs['stride']

        # Initialize low rank metric A and B
        # Depthwise Convolution [in_ch, 1, k, k]
        # self.lora_A = nn.Conv2d(in_channels, in_channels, kernel_size, self.stride, padding=self.padding, groups=in_channels)
        # Pointwise Convolution [out_ch, in_ch, 1, 1]
        self.lora_B = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # self.lora_B = nn.Conv2d(r, out_channels, kernel_size=1, group=1, bias=False)

        # print(f"A {self.lora_A.weight.shape}")
        # print(f"B {self.lora_B.weight.shape}")        

        # For implementation time, combine AB to original convolution
        self.merge = False

        # Pruning mask
        self.is_prune = False
        # self.prune_mask = nn.Parameter(torch.ones(self.conv.weight.shape), requires_grad=False)
        self.lora_mask = nn.Parameter(torch.ones(self.conv.weight.shape), requires_grad=False)
    
    def reset_parameters(self):
        self.lora_A.reset_parameters()
        self.lora_B.reset_parameters()

    def merge_weights(self):
        # weight_A = self.lora_A.weight  # Shape: (r, in_channels, 1, 1)
        # print(f"A {self.lora_A.weight.shape}")
        # weight_B = self.lora_B.weight  # Shape: (out_channels, r, kernel_size, kernel_size)
        # print(f"B {self.lora_B.weight.shape}")
        # merge A and B to original convolution
        # Reshape to match the desired output convolutional kernel shape [out_ch, in_ch, k, k]
        self.conv.weight.data = torch.einsum('oijk, iabc -> oibc', self.lora_B.weight, self.lora_A.weight)
        # merged_weight = torch.einsum('ijkl,mikl->mjkl', weight_A, weight_B) #torch.mul(weight_A, weight_B).view(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        # Ensure the merged weight is contiguous in memory
        # merged_weight = merged_weight.contiguous()

        # Assign merged weight to conv weight as a Parameter
        # self.conv.weight = nn.Parameter(merged_weight)
        # self.conv.weight.data = merged_weight
        # lora_A_weight = self.lora_A.weight.view(self.lora_A.out_channels, -1)
        # print(f"A {lora_A_weight.shape}")
        # lora_B_weight = self.lora_B.weight.view(self.lora_B.out_channels, -1)
        # print(f"B {lora_B_weight.shape}")
        # self.conv.weight.data = (lora_B_weight @ lora_A_weight).view(self.conv.weight.shape) * self.scaling

        # if self.lora_B.bias is not None:
        #     self.conv.bias = self.lora_B.bias
    
    def prune_weights(self, mask=None):
        self.is_prune = True
        # self.lora_mask = nn.Parameter(mask)
    
    def forward(self, x):
        # For Training: 
        if self.training:
            # print('Train')
            # Prune = True, Merge = False
            if self.is_prune:
                # print(f"X_OUT {x_out.shape}")
                # print(f"Mask {self.prune_mask.shape}")
                # Use einsum to multiply with prune_mask without changing its shape
                # combined_weight = torch.einsum('ijkl,mikl->mjkl', self.lora_B.weight, self.lora_A.weight)
                # self.conv.weight.data = torch.einsum('oijk, iabc -> oibc', self.lora_B.weight, self.lora_A.weight)
                # self.conv.weight.data = self.conv.weight.data * self.prune_mask
                # print(f"A {self.lora_A.weight.shape}")
                # print(f"B {self.lora_B.weight.shape}")
                # print(f"B {combined_weight.shape}")
                # print(f"O {self.conv.weight.shape}")
                # x_out = self.conv(x)
                x_out = self.conv._conv_forward(
                            x, 
                            torch.einsum('oijk, iabc -> oibc', self.lora_B.weight, self.lora_A.weight) * self.lora_mask,
                            self.conv.bias
                        )
            else:
                # Prune = False, Merge = False
                x_reduced = self.lora_A(x)
                # print(x_reduced.shape)
                x_out = self.lora_B(x_reduced)
        # For Testing:
        else:
            # Combin weight to convolution
            self.remove()
            # print(f"conv {self.conv.weight.shape}")
            x_out = self.conv(x)
        return x_out
    
    def remove(self):
        self.merge_weights()
        if self.is_prune:
            self.conv.weight.data = self.conv.weight.data * self.lora_mask
        self.merge = True


# # ### AB Convolution layer
# class ABConv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs):
#         super(ABConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
#         LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
#         assert isinstance(kernel_size, int)

#         self.kernel = kernel_size
#         # self.lora_mask = nn.Parameter(torch.ones(self.conv.weight.shape), requires_grad=False)

#         if r > 0:
#             self.lora_A = nn.Parameter(
#                 self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size * kernel_size))
#             )
#             self.lora_B = nn.Parameter(
#               self.conv.weight.new_zeros((out_channels//self.conv.groups, r*kernel_size))
#             )
#             self.scaling = self.lora_alpha / self.r
#             # Freezing the pre-trained weight matrix
#             self.conv.weight.requires_grad = False
#             # self.is_prune = True
#         self.reset_parameters()
#         # print(self.lora_A)
#         # print(self.lora_B)
#         self.merged = False

#     def reset_parameters(self):
#         self.conv.reset_parameters()
#         if hasattr(self, 'lora_A'):
#             # initialize A the same way as the default for nn.Linear and B to zero
#             nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
#             nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))

#     def train(self, mode=True):
#         super(ABConv, self).train(mode)
#         # print(self.conv.weight.shape)
#         if mode: # Train
#             self.merged = False
#         else: # Test merge weights
#             self.conv.weight.data = (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
#             self.merged = True
#         # print(self.merged)

#     def forward(self, x):
#         if self.r > 0 and not self.merged: # Train
#             # print('Train')
#             # if hasattr(self, 'lora_mask'):
#             return self.conv._conv_forward(
#                 x, 
#                 (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
#                 self.conv.bias
#             )

#         else: # Test
#             # print(Test)
#             # if hasattr(self, 'lora_mask'):
#             return self.conv(x)
    # def forward(self, x):
    #     # results=self.conv(x)
    #     # print(f"results{results.shape}")
    #     if self.r > 0 and not self.merged: # Train
    #         # batch_size, in_channels, height, width = x.shape
    #         # print(f"x: {x.shape}")
    #         # print(f"A: {self.lora_A.shape}")
    #         # print(f"B: {self.lora_B.shape}")
    #         # print(f"conv: {self.conv.weight.shape}")
    #         # print(self.conv.kernel_size)
    #         # print(self.conv.padding)
    #         # print(self.conv.stride)
    #         # print(self.conv.dilation)
    #         # Step 1: 对输入 x 执行 im2col 展开操作，使其可以与 lora_A 进行矩阵乘法
    #         x_unfold = F.unfold(x, kernel_size=self.kernel, dilation=self.conv.dilation, padding=self.conv.padding, stride=self.conv.stride)
    #         # print(f"unfold: {x_unfold.shape}")
    #         # Step 2: 输入与 lora_A 进行矩阵运算
    #         x_unfold = x_unfold.permute(1, 0, 2).contiguous().view(x.shape[1] * self.kernel * self.kernel, -1)
    #         x_A = torch.matmul(self.lora_A, x_unfold)
    #         # print(f"A: {x_A.shape}")
            
    #         # Step 3: 将与 lora_B 进行矩阵运算，类似于卷积的第二步
    #         x_B = torch.matmul(self.lora_B, x_A)
    #         # print(f"B: {x_B.shape}")
            
    #         # Step 4: 将矩阵运算结果重新折叠回卷积的输出形状
    #         output_height = (x.shape[2] + 2 * self.conv.padding[0] - self.conv.kernel_size[0]) // self.conv.stride[0] + 1
    #         output_width = (x.shape[3] + 2 * self.conv.padding[1] - self.conv.kernel_size[1]) // self.conv.stride[1] + 1
    #         x_out = x_B.view(x.shape[0], self.conv.out_channels, output_height, output_width)
    #         # print(f"x: {x_out.shape}")
    #         # Step 5: 乘以 scaling 并返回结果
    #         return x_out * self.scaling
    #     else: # Test
    #         # print(Test)
    #         return self.conv(x)