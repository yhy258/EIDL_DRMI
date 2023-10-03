import torch
import matplotlib.pyplot as plt
import models
import schedulers
from utils import load_models
from data.datasets import *
import pyiqa

def save_image_bunch_npy(save_path, datas): # shape : BS, C, H, W
    if isinstance(datas, torch.Tensor):
        datas = datas.cpu().numpy()
    np.save(save_path, datas)

def pnsr_ssim_eval(config, Xs, Ys, device):

    # pyiqa
    x, y = Xs, Ys
    
    if config.dataset_normalization:
        x, y = (x+1)/2, (y+1)/2
    
    psnr_metric = pyiqa.create_metric('psnr').to(device)
    ssim_metric = pyiqa.create_metric('ssim').to(device)
    lpips_metric = pyiqa.create_metric('lpips', device=device)
    psnr = psnr_metric(x, y)
    ssim = ssim_metric(x, y)
    lpips = lpips_metric(x, y)
    return torch.mean(psnr).item(), torch.mean(ssim).item(), torch.mean(lpips).item()

"""
    사전학습된 모델로 test 이미지 쌍을 만들어준다.
"""
def prepare_and_save_images(config, dataloader, device):

    device = f"cuda:{config.cuda_num}"
    model_type = config.model_type
    if model_type == "SFNet":
        config.mode = ["test", "CSD"] 
    model = models.create_model(config, model_type)
    model = model.to(device)
    model.eval()
    
    disc = None
    disc_fft = None
    dis_opt = None
    
    if config.adversarial or config.fourier_adversarial :
        from models.discriminator import Discriminator
        if config.adversarial and config.fourier_adversarial:
            disc = Discriminator().to(device)
            disc_fft = Discriminator(in_channels=6).to(device)
            dis_opt = torch.optim.Adam(params = (list(disc.parameters()) + list(disc_fft.parameters())), lr=config.lr, betas=[0.0, 0.9])
        else:
            if config.adversarial:
                disc = Discriminator().to(device)
                dis_opt = torch.optim.Adam(params=disc.parameters(), lr=config.lr, betas=[0.0, 0.9])
                disc.eval()

            if config.fourier_adversarial:
                disc_fft = Discriminator(in_channels=6).to(device)
                dis_opt = torch.optim.Adam(params=disc_fft.parameters(), lr=config.lr, betas=[0.0, 0.9])
                disc_fft.eval()

    if config.optim_type == "AdamW":
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.lr, betas=[0.9, 0.9])
    else :
        optimizer = torch.optim.Adam(params=model.parameters(), lr=config.lr, betas=[0.9, 0.9])
        
    if config.scheduler == "CosineAnnealingRestartCyclicLR" :
        scheduler = schedulers.CosineAnnealingRestartCyclicLR(optimizer, config.periods, config.restart_weights, config.eta_mins, last_epoch=-1)
        
    else :
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.iteration, eta_min=1e-7, last_epoch=-1 , verbose=False) # per iteration
 
    if config.scheduler == "GradualWarmupScheduler":
        warmup_epochs = 3
        scheduler = schedulers.GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler)
        scheduler.step()
    
    # load

    this_iter, loss_dict = load_models(config,
            model,
            optimizer,
            scheduler,
            disc=disc,
            disc_fft=disc_fft,
            dis_opt=dis_opt,
            epoch=config.eval_ep)


    gt_samples = []
    recon_samples = []
    print(f"Information : Model Type : {config.model_type} , Loss Info : {config.loss_types} , Adv : {config.adversarial} , Fourier Adv : {config.fourier_adversarial}")

    ep = int(config.eval_ep)

    print("LOG : Save Image Data.npy")
    with torch.no_grad():
        for i, data in enumerate(dataloader):

            if config.coord_info:
                x, y, pe = data
                x, y, pe = x.to(device), y.to(device), pe.to(device)
                x_prime = model([y, pe])
            
            else:
                x, y = data
                x, y = x.to(device), y.to(device)
                x_prime = model(y)

            if not isinstance(x_prime, list):
                x_prime = [x_prime]
            
            if config.dataset_normalization:
                x_prime = torch.clamp(x_prime[-1], -1., 1.)
            else:
                x_prime = torch.clamp(x_prime[-1], 0, 1.)


            gt_samples.append(x.detach().cpu())
            recon_samples.append(x_prime.detach().cpu())

    gt_samples = torch.cat(gt_samples, dim=0)
    recon_samples = torch.cat(recon_samples, dim=0)
    print(gt_samples.shape, recon_samples.shape)

    img_save_root = os.path.join(config.image_save_path, config.model_type)
    os.makedirs(img_save_root, exist_ok=True) 
    if config.adversarial and config.fourier_adversarial:
        gt_name = f"adv_fourier_adv_gt_imgs_{ep}"
        recon_name = f"adv_fourier_adv_recon_imgs_{ep}"
    elif config.adversarial:
        gt_name = f"adv_gt_imgs_{ep}"
        recon_name = f"adv_recon_imgs_{ep}"
    elif config.fourier_adversarial:
        gt_name = f"fourier_adv_gt_imgs_{ep}"
        recon_name = f"fourier_adv_recon_imgs_{ep}"
    else :
        gt_name = f"gt_imgs_{ep}.npy"
        recon_name = f"recon_imgs_{ep}.npy"  
    if config.coord_info:
        gt_name = "coord_" + gt_name
        recon_name = "coord_" + recon_name

    gt_path = os.path.join(img_save_root, gt_name)
    recon_path = os.path.join(img_save_root, recon_name)
    save_image_bunch_npy(gt_path, gt_samples)
    save_image_bunch_npy(recon_path, recon_samples)

    return gt_samples, recon_samples, loss_dict

def visualize(config, gt_samples, recon_samples, loss_info, num=4):
    x = gt_samples.cpu().permute(0, 2, 3, 1)[:num]
    x_prime = recon_samples.cpu().permute(0, 2, 3, 1)[:num]
    
    if config.dataset_normalization:
        x_prime = (x_prime + 1) / 2
        x = (x + 1) / 2
    
    # subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(num+1, 2)
    for i in range(num):
        axarr[i, 0].imshow(x[i])
        axarr[i, 0].grid(False)
        axarr[i, 0].axis(False)

        axarr[i, 1].imshow(x_prime[i])
        axarr[i, 1].grid(False)
        axarr[i, 1].axis(False)



    axarr[-1, 0].plot(loss_info["Total Loss"])
    axarr[-1, 0].title.set_text('Total Loss Plot')
    axarr[-1, 1].plot(loss_info["PSNR Eval"])
    axarr[-1, 1].title.set_text('PSNR Eval Plot')


    plt.show()
    
    
    

def evaluation(config):
    device = f'cuda:{config.cuda_num}' if torch.cuda.is_available() else 'cpu'
    
    config.dataset_normalization=False
        
    if config.adversarial or config.fourier_adversarial :
        config.dataset_normalization = True

    if config.data_mode == "lmdb":
        dataset = LMDBMetaLensPair(train=False, normalization=config.dataset_normalization, coord_info=config.coord_info)
    else :
        dataset = MetaLensPair(train=False, normalization=config.dataset_normalization, coord_info=config.coord_info)
        
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=False)

    gt_samples, recon_samples, loss_dict = prepare_and_save_images(config, dataloader, device)

    pnsr, ssim, lpips = pnsr_ssim_eval(config, gt_samples, recon_samples, device)
    print(f"-- PSNR : {pnsr} \t SSIM : {ssim} \t LPIPS : {lpips} --")
    
    # visualization:
    visualize(config, gt_samples, recon_samples, loss_dict, num=4)