from train import train
if __name__ == "__main__":
    from configure import DRMI_Config
    print("Train--")
    config = DRMI_Config()
    config.cuda_num = 3
    config.start_epoch = 0
    config.coord_info = True
    if hasattr(config, "coord_info"):
        if config.coord_info:
            config.in_channels = 5
    config.loss_types = ["PSNRLoss"]
    config.loss_weights = [1.0]

    config.fourier_adversarial = True
    config.gan_weights = {"fft_adv" :0.5}


    print("Config Information")
    print(f"Model Type : {config.model_type}\
            \nData Mode : {config.data_mode}\
            \nCoord Info : {config.coord_info}\
            \nEntire Iteration : {config.iteration}\
            \nAdversarial Training : {config.adversarial}\
            \nFourier Adversarial Training : {config.fourier_adversarial}\
            \nOptimizer Type : {config.optim_type}\
            \nScheduler Type : {config.scheduler}\
            \nLoss Type : {config.loss_types}")
    train(config)
