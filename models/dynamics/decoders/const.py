APP_LSD = 2
DYN_LSD = 2
PROC_PATH_LENGTH = 19

'''
Number of paramaters for decoder (using torch.summary)
    Sequential: 110,108
    Hidden State: 33,484
    Recurrent: 51,194

Variable Conventions:
    h: hidden state
    t: time
    w: something in dynamics latent space
    z: something in appearance latent space
'''