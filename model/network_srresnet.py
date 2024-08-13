import math
import torch.nn as nn
import model.basicblock as B


'''
# ===================
# srresnet
# ===================
'''


class SRResNet1(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=16, upscale=2, act_mode='R', upsample_mode='upconv'):
        super(SRResNet1, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        m_head = B.conv(in_nc, nc, mode='C')

        m_body = [B.ResBlock(nc, nc, mode='C'+act_mode+'C') for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            m_uper = upsample_block(nc, nc, mode='3'+act_mode)
        else:
            m_uper = [upsample_block(nc, nc, mode='2'+act_mode) for _ in range(n_upscale)]

        H_conv0 = B.conv(nc, nc, mode='C'+act_mode)
        H_conv1 = B.conv(nc, out_nc, bias=False, mode='C')
        H_conv2 = B.conv(out_nc,out_nc,bias=False,mode='C')
        m_tail = B.sequential(H_conv0, H_conv1)

        self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper, m_tail)
        self.final = H_conv2
    def forward(self, x):
        x = self.model(x)
        y = self.final(x)
        return x,y

class SRResNet2(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=16, upscale=2, act_mode='R', upsample_mode='upconv'):
        super(SRResNet2, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        m_head = B.conv(in_nc, nc, mode='C')

        m_body = [B.ResBlock(nc, nc, mode='C' + act_mode + 'C') for _ in range(nb)]
        m_body.append(B.conv(nc, nc, mode='C'))

        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            m_uper = upsample_block(nc, nc, mode='3' + act_mode)
        else:
            m_uper = [upsample_block(nc, nc, mode='2' + act_mode) for _ in range(n_upscale)]

        H_conv0 = B.conv(nc, nc, mode='C' + act_mode)
        H_conv1 = B.conv(nc, out_nc, bias=False, mode='C')
        H_conv2 = B.conv(out_nc,out_nc,bias=False,mode='C')
        m_tail  = B.sequential(H_conv0, H_conv1)

        self.model = B.sequential(m_head, B.ShortcutBlock(B.sequential(*m_body)), *m_uper, m_tail)
        self.final = H_conv2

    def forward(self, x):
        x = self.model(x)
        y = self.final(x)
        return x,y
