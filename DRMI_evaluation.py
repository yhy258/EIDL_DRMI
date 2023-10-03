#%%
from evaluation import evaluation

if __name__ == "__main__":
    from configure import DRMI_Config
    config = DRMI_Config()
    config.model_type = "DRMILocal"
    config.cuda_num = 3
    config.eval_ep = 24
    config.coord_info = True
    if hasattr(config, "coord_info"):
        if config.coord_info:
            config.in_channels = 5
            
    config.loss_types = ["PSNRLoss", "FrequencyDivLoss"]
    config.loss_weights = [1.0, 0.5]

#     config.green_skip = True
    config.fourier_adversarial = True
    config.gan_weights = {"fft_adv" :0.5}

    config.var_cal = False # weight :0.1
    config.adaptive_weighting = False
    # config.fourier_adversarial= True
    # config.adversarial= True
    # config.gan_weights = {"fft_adv" :0.5}

    evaluation(config)
# %%
