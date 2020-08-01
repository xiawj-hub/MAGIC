import torch
import torch.nn as nn
from torch.autograd import Function
import torchvision.models as models
import ctlib
import graph_laplacian

class adj_weight(Function):
    def __init__(self, k=9):
        self.k = k

    def forward(self, x):
        return graph_laplacian.forward(x, self.k)

class prj_module(nn.Module):
    def __init__(self, options):
        super(prj_module, self).__init__()  
        self.weight = nn.Parameter(torch.Tensor(1))
        self.options = nn.Parameter(options, requires_grad=False)
        
    def forward(self, input_data, proj):
        return prj_fun.apply(input_data, self.weight, proj, self.options)

class prj_fun(Function):
    @staticmethod
    def forward(self, input_data, weight, proj, options):
        temp = ctlib.projection(input_data, options) - proj
        intervening_res = ctlib.backprojection(temp, options)
        self.save_for_backward(intervening_res, weight, options)
        out = input_data - weight * intervening_res
        return out

    @staticmethod
    def backward(self, grad_output):
        intervening_res, weight, options = self.saved_tensors
        temp = ctlib.projection(grad_output, options)
        temp = ctlib.backprojection(temp, options)
        grad_input = grad_output - weight * temp
        temp = intervening_res * grad_output
        grad_weight = - temp.sum().view(-1)
        return grad_input, grad_weight, None, None

class gcn_module(nn.Module):
    def __init__(self, in_fea, out_fea):
        super(gcn_module, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_fea, out_fea))
        self.bias = nn.Parameter(torch.FloatTensor(out_fea))

    def forward(self, x, adj):
        t = x.view(-1, x.size(2))
        support = torch.mm(t, self.weight)
        support = support.view(x.size(0), x.size(1), -1)
        out = torch.zeros_like(support)
        for i in range(x.size(0)):
            out[i] = torch.mm(adj[i], support[i])
        out = out + self.bias
        return out

class IterBlock(nn.Module):
    def __init__(self, options):
        super(IterBlock, self).__init__()
        self.block1 = prj_module(options)
        self.block2 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )
        self.block3 = gcn_module(36, 64)
        self.block4 = gcn_module(64, 36)
        self.image2patch = image2patch()
        self.patch2image = patch2image()
        self.relu = nn.ReLU(inplace=True)      

    def forward(self, input_data, proj, adj):
        tmp1 = self.block1(input_data, proj)
        tmp2 = self.block2(input_data)
        patch = self.image2patch(input_data)
        tmp3 = self.relu(self.block3(patch, adj))
        tmp3 = self.block4(tmp3, adj)
        tmp3 = self.patch2image(tmp3)
        output = tmp1 + tmp2 + tmp3
        output = self.relu(output)
        return output

class GCN_Learn(nn.Module):
    def __init__(self, block_num, **kwargs):
        super(GCN_Learn, self).__init__()
        views = kwargs['views']
        dets = kwargs['dets']
        width = kwargs['width']
        height = kwargs['height']
        dImg = kwargs['dImg']
        dDet = kwargs['dDet']
        dAng = kwargs['dAng']
        s2r = kwargs['s2r']
        d2r = kwargs['d2r']
        binshift = kwargs['binshift']
        options = torch.Tensor([views, dets, width, height, dImg, dDet, dAng, s2r, d2r, binshift])
        self.block1 = nn.ModuleList([IterBlock(options) for i in range(int(block_num/2))])
        self.block2 = nn.ModuleList([IterBlock(options) for i in range(int(block_num/2))])
        self.image2patch = image2patch()
        self.adj_weight = adj_weight()
    
    def forward(self, input_data, proj):
        x = input_data
        patch1 = self.image2patch(x)
        adj1 = []
        for i in range(input_data.size(0)):
            adj1.append(self.adj_weight.forward(patch1[i]))
        for index, module in enumerate(self.block1):
            x = module(x, proj, adj1)
        adj2 = []
        patch2 = self.image2patch(x)
        for i in range(input_data.size(0)):
            adj2.append(self.adj_weight.forward(patch2[i]))
        for index, module in enumerate(self.block2):
            x = module(x, proj, adj2)
        return x

class image2patch(nn.Module):
    def __init__(self, image_size=256, psize=6, stride=2):
        super(image2patch, self).__init__()
        window_size = image_size + 1 - psize
        mask = torch.arange(window_size*window_size)
        mask = mask.view(window_size, window_size)
        cur = torch.arange(0, window_size, stride)
        if not cur[-1] == window_size - 1:
            cur = torch.cat((cur, torch.LongTensor([window_size - 1])))
        mask = mask[cur,:]
        mask = mask[:,cur]
        self.mask = mask.view(-1)
        self.sizes = torch.LongTensor([psize, window_size, image_size])

    def patch_size(self):
        return self.mask.size(0), self.sizes[0]**2

    def forward(self, input_data):
        return patch_tf.apply(input_data, self.mask, self.sizes)

class patch2image(nn.Module):
    def __init__(self, image_size=256, psize=6, stride=2):
        super(patch2image, self).__init__()
        window_size = image_size + 1 - psize
        mask = torch.arange(window_size*window_size)
        mask = mask.view(window_size, window_size)
        cur = torch.arange(0, window_size, stride)
        if not cur[-1] == window_size - 1:
            cur = torch.cat((cur, torch.LongTensor([window_size - 1])))
        mask = mask[cur,:]
        mask = mask[:,cur]
        self.mask = mask.view(-1)
        self.sizes = torch.LongTensor([psize, window_size, image_size])
        self.ave_mask = ave_mask_com(self.mask, self.sizes)

    def forward(self, input_data):
        return image_tf.apply(input_data, self.mask, self.sizes, self.ave_mask)

def ave_mask_com(mask, sizes):
    ave_mask = torch.zeros(sizes[2], sizes[2])
    patch_set = torch.zeros(sizes[1]**2, sizes[0]**2)
    patch_set[mask] = 1.0
    for i in range(sizes[0]):
        for j in range(sizes[0]):
            index = i * sizes[0]  + j
            temp = patch_set[:,index]
            temp = temp.view(sizes[1], sizes[1])
            ave_mask[i:i+sizes[1],j:j+sizes[1]] = \
                ave_mask[i:i+sizes[1],j:j+sizes[1]] + temp
    ave_mask = ave_mask.view(1, 1, sizes[2], sizes[2])
    return ave_mask

def to_patch(x, mask, sizes, ave_mask=None, mode='sum'):
    batch_size = x.size(0)
    patch_set = torch.zeros(batch_size, sizes[1]**2, sizes[0]**2, device=x.device)
    if mode == 'ave':
        x = x / ave_mask
    for i in range(sizes[0]):
        for j in range(sizes[0]):
            index = i * sizes[0] + j
            temp = x[:,:,i:i+sizes[1],j:j+sizes[1]]
            temp = temp.contiguous().view(batch_size, -1)
            patch_set[:,:,index] = temp
    output = patch_set[:,mask]
    return output

def to_image(x, mask, sizes, ave_mask=None, mode='sum'):
    batch_size = x.size(0)
    patch_set = torch.zeros(batch_size, sizes[1]**2, sizes[0]**2, device=x.device) 
    output= torch.zeros(batch_size, 1, sizes[2], sizes[2], device=x.device)
    patch_set[:,mask] = x
    for i in range(sizes[0]):
        for j in range(sizes[0]):
            index = i * sizes[0]  + j
            temp = patch_set[:,:,index]
            temp = temp.view(batch_size, 1, sizes[1], sizes[1])
            output[:,:,i:i+sizes[1],j:j+sizes[1]] = \
                output[:,:,i:i+sizes[1],j:j+sizes[1]] + temp
    if mode == 'ave':
        output = output / ave_mask
    return output

class patch_tf(Function):
    @staticmethod
    def forward(self, input_data, mask, sizes):
        self.save_for_backward(mask, sizes)
        output = to_patch(input_data, mask, sizes)     
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = None
        mask, sizes = self.saved_tensors
        grad_input = to_image(grad_output, mask, sizes) 
        return grad_input, None, None

class image_tf(Function):
    @staticmethod
    def forward(self, input_data, mask, sizes, ave_mask):
        ave_mask = ave_mask.to(input_data.device)
        self.save_for_backward(mask, sizes, ave_mask)
        output = to_image(input_data, mask, sizes, ave_mask=ave_mask, mode='ave')
        return output
    
    @staticmethod
    def backward(self, grad_output):
        grad_input = None
        mask, sizes, ave_mask = self.saved_tensors
        grad_input = to_patch(grad_output, mask, sizes, ave_mask=ave_mask, mode='ave')
        return grad_input, None, None, None
