class defaultconfig(object):
    attention_model='Generator'
    rdn1='RDN1'
    rdn2='RDN2'
    gather_module='gather_module'
    gather_module1='gather_module1'
    gather_module2='gather_module2'
    gather_module3='gather_module3'
    gather_module4='gather_module4'
    discriminator='discriminator'
    rdn1='RDN1'
    rdn2='RDN2'
    complement='complement'
    reconstruct='reconstruct'
    train_hr='./dataset/sampleval100'
    train_hr1='./dataset/train'
    train_attention='./dataset/gather_low'
    loss='L1_loss'
    at_number=10
    first_channel=3
    in_channel=32
    kernal_size=3
    layer_number=3
    rdb_number=10
    out_channel=32
    batch_size=2
    save_img_dir="./"
    model_path='./checkpoint_x2/net_parameters499'
