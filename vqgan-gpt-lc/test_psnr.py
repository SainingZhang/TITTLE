import torch
import torch.nn.functional as F
import numpy as np 
from PIL import Image
import lpips as lps
from lpips.pretrained_networks import alexnet
from torchvision import models as tv

class local_alexnet(alexnet):
    def __init__(self, path):
        super().__init__(requires_grad=False, pretrained=False)
        tv_alexnet = tv.alexnet(pretrained=False)
        tv_alexnet.load_state_dict(torch.load(path))
        alexnet_pretrained_features = tv_alexnet.features
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        for param in self.parameters():
            param.requires_grad = False

lpips_loss = lps.LPIPS(net='alex',pnet_rand=True)
lpips_loss.net = local_alexnet('/cache/data/alexnet-owt-7be5be79.pth')
lpips_loss = lpips_loss.cuda()

def compute_psnr(recon, ori):
    x = torch.from_numpy(np.array(Image.open(ori))).cuda().float()
    y = torch.from_numpy(np.array(Image.open(recon))).cuda().float()
    mse = F.mse_loss(x, y)
    psnr = 20 * torch.log10(torch.Tensor([255.0]).to(x.device)) - 10 * torch.log10(mse)
    x = x/255
    y = y/255
    x-=1
    y-=1
    #print(x.shape)
    x = x.transpose(2, 0)
    y = y.transpose(2, 0)
    x = x[:3]
    y = y[:3]
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    #print(x.shape)
    #print(x[3])
    # y = y.transpose(1, 2, 0)
    lpips_score = lpips_loss(y.clamp(-1,1), x.clamp(-1,1))
    #lpips_score = 0
    return psnr, lpips_score

total = 0
lpips_total = 0.0
for i in range(1000):
    cur_psnr, cur_lpips = compute_psnr(f"/cache/log_eval_recons/vqgan_lc_100K_f16_SD3_0923_2l/recons/{i}.png", f"/cache/log_eval_recons/vqgan_lc_100K_f16_SD3_0923_2l/recons/real{i}.png")
    #print(cur_psnr)
    total+=cur_psnr
    #print(cur_lpips.sum().item())
    lpips_total+=cur_lpips.sum().item()

print(total/1000)
print(lpips_total/1000)
