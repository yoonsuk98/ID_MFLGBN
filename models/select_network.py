import functools
import torch
from torch.nn import init


"""
# --------------------------------------------
# select the network of G, D and F
# --------------------------------------------
"""


# --------------------------------------------
# Generator, netG, G
# --------------------------------------------
def define_G(opt):
    opt_net = opt['netG']
    net_type = opt_net['net_type']

    '''
    # ----------------------------------------
    # ↓↓↓↓↓↓↓↓↓↓↓↓↓ MY IR task ↓↓↓↓↓↓↓↓↓↓↓↓↓↓
    # ----------------------------------------
    '''

    if net_type == 'PROPOSED':
        from models.network_PROPOSED import RDDA as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   feature_channels=opt_net['n_feats'],
                   patch_size=opt['datasets']['train']['H_size'],
                   norm_type = opt_net['LayerNorm'],
                   num_heads=opt_net['num_heads'],
                   ffn_expansion_factor=opt_net['ffn_expansion_factor']
                   ) 
    # ----------------------------------------
    # DRANet
    # ----------------------------------------

    elif net_type == 'dranet':
        from models.network_dranet import DRANet as net
        netG = net(in_nc=opt_net['in_nc'],
                   out_nc=opt_net['out_nc'],
                   nc=opt_net['nc'],
                   bias=opt_net['bias'],
                   ) 
            

    # ----------------------------------------
    # SwinIR
    # ----------------------------------------
    elif net_type == 'swinir':
        from models.network_swinir import SwinIR as net
        netG = net(upscale=opt_net['upscale'],
                   in_chans=opt_net['in_chans'],
                   img_size=opt['datasets']['train']['H_size'],
                   window_size=opt_net['window_size'],
                   img_range=opt_net['img_range'],
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'],
                   resi_connection=opt_net['resi_connection'])

    
    
    # ----------------------------------------
    # QformerID
    # ----------------------------------------
    elif net_type == 'QformerID':
            from models.network_qformerid import QformerID as net
            netG = net(upscale=opt_net['upscale'],
                    in_chans=4,
                    img_size=opt_net['img_size'],
                    window_size=opt_net['window_size'],
                    img_range=opt_net['img_range'],
                    depths=opt_net['depths'],
                    embed_dim=opt_net['embed_dim'],
                    num_heads=opt_net['num_heads'],
                    mlp_ratio=opt_net['mlp_ratio'],
                    upsampler=opt_net['upsampler'],
                    resi_connection=opt_net['resi_connection'])


    # ----------------------------------------
    # Restormer
    # ----------------------------------------
    elif net_type == 'restormer':
            from models.network_restormer import Restormer as net
            
            netG = net(
                    inp_channels=opt_net['in_chans'],
                    out_channels=opt_net['out_chans'],
                    dim=opt_net['dim'],
                    num_blocks=opt_net['num_blocks'],
                    num_refinement_blocks=opt_net['num_refinement_blocks'],
                    heads=opt_net['heads'],
                    ffn_expansion_factor=opt_net['ffn_expansion_factor'],
                    bias=opt_net['bias'],
                    LayerNorm_type=opt_net['LayerNorm_type'],
                    dual_pixel_task=opt_net['dual_pixel_task'])

    # ----------------------------------------
    # RAMiT
    # ----------------------------------------

    elif net_type == 'ramit':
            from models.network_ramit import RAMiT as net
            netG = net(in_chans=opt_net['in_chans'],
                    dim=opt_net['dim'],
                    depths=opt_net['depths'],
                    num_heads=opt_net['num_heads'],
                    head_dim=opt_net['head_dim'],
                    chsa_head_ratio=opt_net['chsa_head_ratio'],
                    window_size=opt_net['window_size'],
                    hidden_ratio=opt_net['hidden_ratio'],
                    qkv_bias=opt_net['qkv_bias'],
                    mv_ver=opt_net['mv_ver'],
                    exp_factor=opt_net['exp_factor'],
                    expand_groups=opt_net['expand_groups'],
                    norm_layer=opt_net['norm_layer'],
                    tail_mv=opt_net['tail_mv'],
                    target_mode=opt_net['target_mode'],
                    img_norm=opt_net['img_norm'],
                    attn_drop=opt_net['attn_drop'],
                    proj_drop=opt_net['proj_drop'],
                    drop_path=opt_net['drop_path'],
                    helper=opt_net['helper']
                    ) 

    # ----------------------------------------
    # CAT
    # ----------------------------------------
    elif net_type == 'cat':
        from models.network_cat import CAT_Unet as net
        netG = net(
            img_size=opt['datasets']['train']['H_size'],
            in_chans=opt_net['in_chans'],
            depths=opt_net['depths'],
            split_size_0=opt_net['split_size_0'],
            split_size_1=opt_net['split_size_1'],
            dim=opt_net['dim'],
            num_heads=opt_net['num_heads'],
            mlp_ratio=opt_net['mlp_ratio'],
            num_refinement_blocks=opt_net['num_refinement_blocks'],
            bias=opt_net['bias'],
            dual_pixel_task=opt_net['dual_pixel_task'])
    # ----------------------------------------
    # UFormer
    # ----------------------------------------
    elif net_type == 'uformer':
            from models.network_uformer import Uformer as net
            
            netG = net(
                    img_size=opt['datasets']['train']['H_size'],
                    in_chans=opt_net['in_chans'],
                    dd_in=opt_net['dd_in'],
                    embed_dim=opt_net['embed_dim'],
                    depths=opt_net['depths'],
                    num_heads=opt_net['num_heads'],
                    win_size=opt_net['win_size'],
                    mlp_ratio=opt_net['mlp_ratio'],
                    token_projection=opt_net['token_projection'],
                    token_mlp=opt_net['token_mlp']
                    )


    # ----------------------------------------
    # others
    # ----------------------------------------
    # TODO

    else:
        raise NotImplementedError('netG [{:s}] is not found.'.format(net_type))

    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    if opt['is_train']:
        init_weights(netG,
                     init_type=opt_net['init_type'],
                     init_bn_type=opt_net['init_bn_type'],
                     gain=opt_net['init_gain'])

    return netG


# --------------------------------------------
# Discriminator, netD, D
# --------------------------------------------
def define_D(opt):
    opt_net = opt['netD']
    net_type = opt_net['net_type']

    # ----------------------------------------
    # discriminator_vgg_96
    # ----------------------------------------
    if net_type == 'discriminator_vgg_96':
        from models.network_discriminator import Discriminator_VGG_96 as discriminator
        netD = discriminator(in_nc=opt_net['in_nc'],
                             base_nc=opt_net['base_nc'],
                             ac_type=opt_net['act_mode'])

    # ----------------------------------------
    # discriminator_vgg_128
    # ----------------------------------------
    elif net_type == 'discriminator_vgg_128':
        from models.network_discriminator import Discriminator_VGG_128 as discriminator
        netD = discriminator(in_nc=opt_net['in_nc'],
                             base_nc=opt_net['base_nc'],
                             ac_type=opt_net['act_mode'])

    # ----------------------------------------
    # discriminator_vgg_192
    # ----------------------------------------
    elif net_type == 'discriminator_vgg_192':
        from models.network_discriminator import Discriminator_VGG_192 as discriminator
        netD = discriminator(in_nc=opt_net['in_nc'],
                             base_nc=opt_net['base_nc'],
                             ac_type=opt_net['act_mode'])

    # ----------------------------------------
    # discriminator_vgg_128_SN
    # ----------------------------------------
    elif net_type == 'discriminator_vgg_128_SN':
        from models.network_discriminator import Discriminator_VGG_128_SN as discriminator
        netD = discriminator()

    elif net_type == 'discriminator_patchgan':
        from models.network_discriminator import Discriminator_PatchGAN as discriminator
        netD = discriminator(input_nc=opt_net['in_nc'],
                             ndf=opt_net['base_nc'],
                             n_layers=opt_net['n_layers'],
                             norm_type=opt_net['norm_type'])

    elif net_type == 'discriminator_unet':
        from models.network_discriminator import Discriminator_UNet as discriminator
        netD = discriminator(input_nc=opt_net['in_nc'],
                             ndf=opt_net['base_nc'])

    else:
        raise NotImplementedError('netD [{:s}] is not found.'.format(net_type))

    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    init_weights(netD,
                 init_type=opt_net['init_type'],
                 init_bn_type=opt_net['init_bn_type'],
                 gain=opt_net['init_gain'])

    return netD


# --------------------------------------------
# VGGfeature, netF, F
# --------------------------------------------
def define_F(opt, use_bn=False):
    device = torch.device('cuda' if opt['gpu_ids'] else 'cpu')
    from models.network_feature import VGGFeatureExtractor
    # pytorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = VGGFeatureExtractor(feature_layer=feature_layer,
                               use_bn=use_bn,
                               use_input_norm=True,
                               device=device)
    netF.eval()  # No need to train, but need BP to input
    return netF


"""
# --------------------------------------------
# weights initialization
# --------------------------------------------
"""


def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
    """
    # Kai Zhang, https://github.com/cszn/KAIR
    #
    # Args:
    #   init_type:
    #       default, none: pass init_weights
    #       normal; normal; xavier_normal; xavier_uniform;
    #       kaiming_normal; kaiming_uniform; orthogonal
    #   init_bn_type:
    #       uniform; constant
    #   gain:
    #       0.2
    """

    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:

            if init_type == 'normal':
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'uniform':
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)

            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)

            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)

            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(gain)

            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)

            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))

            if m.bias is not None:
                m.bias.data.zero_()

        elif classname.find('BatchNorm2d') != -1:

            if init_bn_type == 'uniform':  # preferred
                if m.affine:
                    init.uniform_(m.weight.data, 0.1, 1.0)
                    init.constant_(m.bias.data, 0.0)
            elif init_bn_type == 'constant':
                if m.affine:
                    init.constant_(m.weight.data, 1.0)
                    init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))

    if init_type not in ['default', 'none']:

        print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))
        fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
        net.apply(fn)
    else:
        print('Pass this initialization! Initialization was done during network definition!')
