import os
import torch


def save_models(config,
                model,
                optimizer,
                scheduler,
                disc=None,
                disc_fft=None,
                dis_opt=None,
                loss_info=None,
                epoch=0,
                this_iter=0):
    
    save_root = os.path.join(config.save_root, config.model_type)
    if "SFFreqeuncyLoss" in config.loss_types or "FocalFrequencyLoss" in config.loss_types :
        save_root = os.path.join(save_root, "freq_loss")
    os.makedirs(save_root, exist_ok=True)

    if config.adversarial and config.fourier_adversarial:
        file_name = f"spatial_fourier_adv_model_{config.gan_weights['adv']}_{config.gan_weights['fft_adv']}_{epoch}.pt"
        if config.coord_info:
            file_name = "coord_" + file_name
        if len(config.loss_types) > 1 :
            for i, lt in enumerate(config.loss_types[1:]):
                loss_name = lt[:-4] + str(config.loss_weights[i+1])
                file_name = loss_name+"_"+file_name
        torch.save({
                "model": model.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "scheduler" : scheduler.state_dict(),
                "spatial_disc" : disc.state_dict(),
                "fft_disc" : disc_fft.state_dict(),
                "dis_opt" : dis_opt.state_dict(),
                "this_iter" : this_iter,
                "loss_info": loss_info
            }, os.path.join(save_root, file_name))
        
    elif config.adversarial:
        file_name = f"spatial_adv_model_{epoch}.pt"
        if config.coord_info:
            file_name = "coord_" + file_name
        if len(config.loss_types) > 1 :
            for i, lt in enumerate(config.loss_types[1:]):
                loss_name = lt[:-4] + str(config.loss_weights[i+1])
                file_name = loss_name+"_"+file_name
        
        torch.save({
                "model": model.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "scheduler" : scheduler.state_dict(),
                "spatial_disc" : disc.state_dict(),
                "dis_opt" : dis_opt.state_dict(),
                "this_iter" : this_iter,
                "loss_info": loss_info
            }, os.path.join(save_root, file_name))
        
    elif config.fourier_adversarial:
        file_name = f"fourier_adv_model_{epoch}.pt"
        if config.coord_info:
            file_name = "coord_" + file_name
        if len(config.loss_types) > 1 :
            for i, lt in enumerate(config.loss_types[1:]):
                loss_name = lt[:-4] + str(config.loss_weights[i+1])
                file_name = loss_name+"_"+file_name
        torch.save({
                "model": model.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "scheduler" : scheduler.state_dict(),
                "fft_disc" : disc_fft.state_dict(),
                "dis_opt" : dis_opt.state_dict(),
                "this_iter" : this_iter,
                "loss_info": loss_info
            }, os.path.join(save_root, file_name))
        
    
    else :
        file_name = f"model_{epoch}.pt"
        if config.coord_info:
            file_name = "coord_" + file_name
        if len(config.loss_types) > 1 :
            for i, lt in enumerate(config.loss_types[1:]):
                loss_name = lt[:-4] + str(config.loss_weights[i])
                file_name = loss_name+"_"+file_name

        torch.save({
                "model": model.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "scheduler" : scheduler.state_dict(),
                "this_iter" : this_iter,
                "loss_info": loss_info
            }, os.path.join(save_root, file_name))
        

def load_models(config,
                model,
                optimizer,
                scheduler,
                disc=None,
                disc_fft=None,
                dis_opt=None,
                epoch=8334):
    
    
    
    if "Local" in config.model_type:
        model_type = config.model_type[:-5]
    else :
        model_type = config.model_type
    save_root = os.path.join(config.save_root, model_type)
    if "SFFreqeuncyLoss" in config.loss_types or "FocalFrequencyLoss" in config.loss_types :
        save_root = os.path.join(save_root, "freq_loss")

    if config.adversarial and config.fourier_adversarial:
        file_name = f"spatial_fourier_adv_model_{config.gan_weights['adv']}_{config.gan_weights['fft_adv']}_{epoch}.pt"
        if config.coord_info:
            file_name = "coord_" + file_name
        if len(config.loss_types) > 1 :
            for i, lt in enumerate(config.loss_types[1:]):
                loss_name = lt[:-4] + str(config.loss_weights[i+1])
                file_name = loss_name+"_"+file_name
            
                
        load_path = os.path.join(save_root, file_name)
        checkpoint = torch.load(load_path, map_location=f"cuda:{config.cuda_num}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        disc.load_state_dict(checkpoint["spatial_disc"])
        disc_fft.load_state_dict(checkpoint["fft_disc"])
        dis_opt.load_state_dict(checkpoint["dis_opt"])
        this_iter = checkpoint["this_iter"]
        loss_info = checkpoint["loss_info"]
        return this_iter, loss_info
        
    elif config.adversarial: 
        file_name = f"spatial_adv_model_{epoch}.pt"
        if config.coord_info:
            file_name = "coord_" + file_name
            
        if len(config.loss_types) > 1 :
            for i, lt in enumerate(config.loss_types[1:]):
                loss_name = lt[:-4] + str(config.loss_weights[i+1])
                file_name = loss_name+"_"+file_name
                
        load_path = os.path.join(save_root, file_name)
        checkpoint = torch.load(load_path, map_location=f"cuda:{config.cuda_num}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        disc.load_state_dict(checkpoint["spatial_disc"])
        dis_opt.load_state_dict(checkpoint["dis_opt"])
        this_iter = checkpoint["this_iter"]
        loss_info = checkpoint["loss_info"]
        return this_iter, loss_info

        
    elif config.fourier_adversarial:
        file_name = f"fourier_adv_model_{epoch}.pt"
        if config.coord_info:
            file_name = "coord_" + file_name

        if len(config.loss_types) > 1 :
            for i, lt in enumerate(config.loss_types[1:]):
                loss_name = lt[:-4] + str(config.loss_weights[i+1])
                file_name = loss_name+"_"+file_name
        
        load_path = os.path.join(save_root, file_name)
        print(load_path)
        checkpoint = torch.load(load_path, map_location=f"cuda:{config.cuda_num}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        disc_fft.load_state_dict(checkpoint["fft_disc"])
        dis_opt.load_state_dict(checkpoint["dis_opt"])
        this_iter = checkpoint["this_iter"]
        loss_info = checkpoint["loss_info"]
        return this_iter, loss_info
        
        
    else :
        file_name = f"model_{epoch}.pt"

        if config.coord_info:
            file_name = "coord_" + file_name
            
        if len(config.loss_types) > 1 :
            for i, lt in enumerate(config.loss_types[1:]):
                loss_name = lt[:-4] + str(config.loss_weights[i+1])
                file_name = loss_name+"_"+file_name


        load_path = os.path.join(save_root, file_name)
        checkpoint = torch.load(load_path, map_location=f"cuda:{config.cuda_num}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        this_iter = checkpoint["this_iter"]
        loss_info = checkpoint["loss_info"]
        return this_iter, loss_info