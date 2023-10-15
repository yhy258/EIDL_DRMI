import os

class DRMI_Config():
    cuda_num = 0
    model_type = "DRMI"
    data_path = ""
    patch_size = 256
    
    # train
    entire_epoch = 9000
    start_epoch = 0
    iteration = 300000
    lr = 3e-4
    batch_size = 16
    progressive_learning = False
    coord_info = True
    
    adversarial = False
    fourier_adversarial = True
    use_grad_clip = True
    
    optim_type = "AdamW"
    scheduler = "CosineAnnealingLR"

    loss_types = ["PSNRLoss"]

    
    # NAFNet's Parameters
    
    in_channels = 3
    width = 32
    middle_blk_num = 12
    enc_blk_nums = [2, 2, 4, 8]
    dec_blk_nums = [2, 2, 2, 2]

    # model save
    save_root = ""   
    save_frequency = 1000
    
    if "Local" in model_type:
        image_save_path = os.path.join(save_root, model_type[:-5])
    else :
        image_save_path = os.path.join(save_root, model_type)
    eval_ep = 7500
    
