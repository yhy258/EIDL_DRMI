# 모든 훈련은 epoch을 세면서 실행한다. 하지만 stop criterion은 Iteration.
# progressive learning scheme은 나중에 넣자 나중에..
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import models
from data.datasets import LMDBMetaLensPair
from data.data_util import get_iter_flag, define_data_instances
from loss import losses
import schedulers
from utils import save_models, load_models

"""Peak Signal to Noise Ratio
img1 and img2 have range [0, 255]"""
class PSNR:
    def __init__(self):
        self.name = "PSNR"

    @torch.no_grad()
    def __call__(self, img1, img2, scaler="01"): # bs, c, h, w
        if scaler == "01":
            img1, img2 = img1 * 255., img2 * 255.
        else :
            img1, img2 = (img1 + 1)/2., (img2 + 1)/2.
            img1, img2 = img1 * 255., img2 * 255.
        
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))
    
def data_scaler(data):
    return ((data - data.min()) / (data.max() - data.min()))*2 - 1

def train(config):
    device = f"cuda:{config.cuda_num}"
    model_type = config.model_type
    H, W = 800, 1280
    dataset_normalization = False
    if model_type == "SFNet":
        config.mode = ["train"]
        
    model = models.create_model(config, model_type)
    model = model.to(device)
    
    disc = None
    disc_fft = None
    dis_opt = None
    
    if config.adversarial or config.fourier_adversarial :
        from models.discriminator import Discriminator
        dataset_normalization = True
        if config.adversarial and config.fourier_adversarial:
            disc = Discriminator().to(device)
            disc_fft = Discriminator(in_channels=6).to(device)
            dis_opt = torch.optim.Adam(params = (list(disc.parameters()) + list(disc_fft.parameters())), lr=config.lr, betas=[0.0, 0.9])
        else :
            if config.adversarial:
                disc = Discriminator().to(device)
                dis_opt = torch.optim.Adam(params=disc.parameters(), lr=config.lr, betas=[0.0, 0.9])

            if config.fourier_adversarial:
                disc_fft = Discriminator(in_channels=6).to(device)
                dis_opt = torch.optim.Adam(params=disc_fft.parameters(), lr=config.lr, betas=[0.0, 0.9])

    if config.optim_type == "AdamW":
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.lr, betas=[0.9, 0.9])
    else :
        optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr, betas=[0.9, 0.9])
    if config.adversarial or config.fourier_adversarial:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr, betas=[0.9, 0.9])
    
        
    if config.scheduler == "CosineAnnealingRestartCyclicLR" :
        scheduler = schedulers.CosineAnnealingRestartCyclicLR(optimizer, config.periods, config.restart_weights, config.eta_mins, last_epoch=-1)
        
    else :
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.iteration, eta_min=1e-7, last_epoch=-1 , verbose=False) # per iteration
 
    if config.scheduler == "GradualWarmupScheduler":
        warmup_epochs = 3
        scheduler = schedulers.GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler)
        scheduler.step()
        
    start_epoch = 1
    this_iter = 0
    loss_dict = None
    
    # resume
    if config.start_epoch > 0 :
        start_epoch = config.start_epoch
        this_iter, loss_dict = load_models(config,
                model,
                optimizer,
                scheduler,
                disc=disc,
                disc_fft=disc_fft,
                dis_opt=dis_opt,                
                epoch=start_epoch)
 
    # define criterions
    criterion_clss = [getattr(losses, lt) for lt in config.loss_types]
    freqdivindex = None
    if hasattr(config, "loss_weights"):
        criterions = [cc(loss_weight=config.loss_weights[i]).to(device) for i, cc in enumerate(criterion_clss)]
    else :
        criterions = [cc().to(device) for cc in criterion_clss]
    for i, c in enumerate(criterions):
        if isinstance(c, losses.PerceptualLoss):
            c.device = device
            print(f"Perceptual Loss Device : {c.device}")
        if isinstance(c, losses.FrequencyDivLoss):
            freqdivindex = i
        
        
    adv_criterions = {}
    if config.adversarial or config.fourier_adversarial:
        criterion_cls = getattr(losses, "AdversarialLoss")
        if config.adversarial:
            adv_criterions["adv"] = criterion_cls(gan_weight=config.gan_weights["adv"]).to(device)    
        if config.fourier_adversarial:
            adv_criterions["fft_adv"] = criterion_cls(gan_weight=config.gan_weights["fft_adv"]).to(device)
        
        
    psnr_eval = PSNR()
    
    # Dataset, DataLoader
    if config.progressive_learning :
        cumsum_iters = np.cumsum(config.iters)
        progressive_index = get_iter_flag(cumsum_iters, this_iter)
        if progressive_index+1 >= len(cumsum_iters):
            next_iter = this_iter
        else:
            next_iter = cumsum_iters[progressive_index+1]

        dataset = LMDBMetaLensPair(patch_size=config.gt_sizes[progressive_index], normalization=dataset_normalization)
        dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=config.mini_batch_sizes[progressive_index], num_workers=8, pin_memory=True)
    else :
        dataset = LMDBMetaLensPair(patch_size=config.patch_size, normalization=dataset_normalization, coord_info=config.coord_info)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    if loss_dict == None:
        loss_dict = {"Total Loss" : [], "PSNR Eval" : []}

    # trainer
    for epoch in range(start_epoch, config.entire_epoch + 1):
        print(f"###### Now Epoch : {epoch} / {config.entire_epoch}")
        psnr_list = []
        loss_list = []
        for data in tqdm(dataloader) :
            if config.progressive_learning:
                if next_iter < this_iter :
                    break


            if config.coord_info:
                x, y, pe = data
                x, y, pe = x.to(device), y.to(device), pe.to(device)
                x_prime = model([y, pe])
            else :
                x, y = data
                x, y = x.to(device), y.to(device)
                x_prime = model(y)            
                
            if not isinstance(x_prime, list):
                x_prime = [x_prime]
                
            this_loss = 0
        
            if dataset_normalization:
                this_psnr = psnr_eval(x_prime[-1], x, "-11")
            else :
                this_psnr = psnr_eval(x_prime[-1], x)

            if config.adversarial or config.fourier_adversarial:
                if config.adversarial and config.fourier_adversarial:
                    x_fft = torch.fft.fft2(x, dim=(-2,-1))
                    x_prime_fft = torch.fft.fft2(x_prime[-1], dim=(-2,-1))
                    if freqdivindex is not None:
                        this_loss += criterions[freqdivindex](x_fft, x_prime_fft, ff_transformed=True)
                    x_fft = torch.cat((data_scaler(x_fft.real), data_scaler(x_fft.imag)), 1)
                    x_prime_fft = torch.cat((data_scaler(x_prime_fft.real), data_scaler(x_prime_fft.imag)), 1)

                    fft_real = disc_fft(x_fft)
                    fft_fake = disc_fft(x_prime_fft)
                    real = disc(x)
                    fake = disc(x_prime[-1])
                    
                    d_loss = (adv_criterions["fft_adv"](fft_fake, fft_real, mode="D") + adv_criterions["adv"](fake, real, mode="D"))/2
                    dis_opt.zero_grad()
                    d_loss.backward(retain_graph=True)
                    dis_opt.step()

                    g_loss = (adv_criterions["fft_adv"](fft_fake, mode="G") + adv_criterions["adv"](fake, mode="G"))/2
                    this_loss += g_loss
            
                else:
                    if config.adversarial:
                        real = disc(x)
                        fake = disc(x_prime[-1])
                        d_loss = adv_criterions["adv"](fake, real, mode='D')
                        dis_opt.zero_grad()
                        d_loss.backward(retain_graph=True)
                        dis_opt.step()
                        g_loss = adv_criterions["adv"](fake, mode="G")
                        this_loss += g_loss
   
                    if config.fourier_adversarial :
                        x_fft = torch.fft.fft2(x, dim=(-2,-1))
                        x_prime_fft = torch.fft.fft2(x_prime[-1], dim=(-2,-1))

                        if freqdivindex is not None:
                            this_loss += criterions[freqdivindex](x_fft, x_prime_fft, ff_transformed=True)

                        x_fft = torch.cat((data_scaler(x_fft.real), data_scaler(x_fft.imag)), 1)
                        x_prime_fft = torch.cat((data_scaler(x_prime_fft.real), data_scaler(x_prime_fft.imag)), 1)
                        
                        real = disc_fft(x_fft)
                        fake = disc_fft(x_prime_fft)
                        
                        d_loss = adv_criterions["fft_adv"](fake, real, mode='D')
                        dis_opt.zero_grad()
                        d_loss.backward(retain_graph=True)
                        dis_opt.step()
                        g_loss = adv_criterions["fft_adv"](fake, mode="G")
                        this_loss += g_loss 
   
            if config.model_type == "SFNet":
                x2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
                x4 = F.interpolate(x, scale_factor=0.25, mode='bilinear')
                xs = [x4, x2, x]
                for j in range(len(x_prime)):
                    for k, criterion in enumerate(criterions) :
                        if k == freqdivindex and config.fourier_adversarial :
                            continue
                        this_loss += criterion(x_prime[j], xs[j])

            else :
                for j in range(len(x_prime)):
                    for k, criterion in enumerate(criterions) :
                        if k == freqdivindex and config.fourier_adversarial :
                            continue
                        if (config.loss_types[k] == "PSNRLoss") and (dataset_normalization):
                            pred, target = (x_prime[j] + 1)/2, (x + 1)/2
                            this_loss += criterion(pred, target)
                        else :
                            this_loss += criterion(x_prime[j], x)

            loss_list.append(this_loss.item())
            psnr_list.append(this_psnr.item())

            optimizer.zero_grad()
            this_loss.backward()
            if config.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            optimizer.step()
            
            if config.scheduler != "GradualWarmupScheduler":
                scheduler.step()
            
            this_iter += 1 
            torch.cuda.empty_cache()
            if this_iter >= config.iteration:
                save_models(config,
                    model,
                    optimizer,
                    scheduler,
                    disc=disc,
                    disc_fft=disc_fft,
                    dis_opt=dis_opt,
                    loss_info=loss_dict,
                    epoch=epoch,
                    this_iter=this_iter)
                break
        
        if config.progressive_learning:
            if next_iter < this_iter :
                progressive_index += 1
                if progressive_index+1 >= len(cumsum_iters):
                    next_iter = this_iter
                else:
                    next_iter = cumsum_iters[progressive_index+1]
                _, dataloader = define_data_instances(config, progressive_index, dataset_normalization)
                torch.cuda.empty_cache()
                continue

        if config.scheduler == "GradualWarmupScheduler": # per epoch
            scheduler.step()
        
        print(f"Information : Model Type : {config.model_type} , Loss Info : {config.loss_types} , Adv : {config.adversarial} , Fourier Adv : {config.fourier_adversarial} , coord_info : {config.coord_info}")
        mean_psnr = np.mean(psnr_list)
        mean_loss = np.mean(loss_list)
        print(f"Total Loss : {mean_loss} \t PSNR : {mean_psnr}")
        if config.adversarial or config.fourier_adversarial:
            print(config.gan_weights)
        print(this_iter)
        
        # dictionary update
        loss_dict["Total Loss"].append(mean_loss)
        loss_dict["PSNR Eval"].append(mean_psnr)
        
        if epoch % config.save_frequency == 0:
            save_models(config,
                model,
                optimizer,
                scheduler,
                disc=disc,
                disc_fft=disc_fft,
                dis_opt=dis_opt,                
                loss_info=loss_dict,
                epoch=epoch,
                this_iter=this_iter)
            
        if this_iter >= config.iteration:
            break
            
        
        
            
        

