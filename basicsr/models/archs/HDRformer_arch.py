import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from basicsr.utils.registry import ARCH_REGISTRY  # 用于模型注册（BasicSR框架）



# ============ 初始化方法 ============

# 无梯度下的截断正态初始化函数
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):  # 正态分布累积分布函数
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    # 检查均值是否在合理范围内，否则发出警告
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # 均匀初始化 + 反误差函数 -> 截断高斯分布
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

# 对外暴露的截断正态初始化接口
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

# 按照变异性缩放规则初始化张量
def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)

    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    # 根据不同分布执行初始化
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")

# LeCun Normal 初始化（适用于激活函数为 tanh 的网络）
def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


# ============ 模块定义 ============

# 预归一化模块：先进行 LayerNorm，再送入后续模块
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)

# GELU 激活函数模块封装
class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

# 卷积封装函数，默认 stride=1，padding=kernel//2
def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)

# 图像列向后平移函数（数据增强/预处理常用）
# 输入：[bs,28,256,310] 输出：[bs,28,256,256]
def shift_back(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    down_sample = 256 // row  # 下采样比例
    step = float(step) / float(down_sample * down_sample)
    out_col = row  # 目标输出列数
    for i in range(nC):
        inputs[:, i, :, :out_col] = \
            inputs[:, i, :, int(step * i):int(step * i) + out_col]
    return inputs[:, :, :, :out_col]


# 亮度估计模块（输入图像，输出特征和估计图）
class Luminance_Estimator(nn.Module):
    def __init__(self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super(Luminance_Estimator, self).__init__()
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # 输入图像 img：[B, 3, H, W]
        # 通道均值
        mean_c = img.mean(dim=1).unsqueeze(1)  # [B, 1, H, W]
        input = torch.cat([img, mean_c], dim=1)  # 拼接出4通道输入

        x_1 = self.conv1(input)           # 通道变换
        lum_fea = self.depth_conv(x_1)   # 深度卷积提取特征
        lum_map = self.conv2(lum_fea)   # 输出亮度图
        return lum_fea, lum_map


# 引导注意力多头模块（以亮度引导注意力机制）
class LG_MSA(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, lum_fea_trans):
        # 输入 x_in: [B, H, W, C]
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)  # 拉平空域

        # 线性映射为 Q/K/V
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)

        lum_attn = lum_fea_trans  # b,h,w,c 光照引导特征

        # rearrange 为 [B, heads, N, D]
        q, k, v, lum_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp, lum_attn.flatten(1, 2)))

        v = v * lum_attn  # 光照特征调制 V

        # 转置 + 归一化 + 计算注意力权重
        q = F.normalize(q.transpose(-2, -1), dim=-1, p=2)
        k = F.normalize(k.transpose(-2, -1), dim=-1, p=2)
        v = v.transpose(-2, -1)

        attn = (k @ q.transpose(-2, -1)) * self.rescale  # 注意力计算并缩放
        attn = attn.softmax(dim=-1)

        x = attn @ v  # 注意力加权输出
        x = x.permute(0, 3, 1, 2).reshape(b, h * w, self.num_heads * self.dim_head)

        out_c = self.proj(x).view(b, h, w, c)  # 输出通道融合
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        out = out_c + out_p  # 加上位置编码
        return out


# 前馈网络（残差FFN）
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


# LGAB：引导注意力块（多层 LG-MSA + FFN）
class LGAB(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                LG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, illu_fea):
        x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
        for (attn, ff) in self.blocks:
            x = attn(x, illu_fea.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
        return x.permute(0, 3, 1, 2)  # BHWC -> BCHW


# 色域网络（残差FFN）
class ChromaMLP(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        out = self.net(x)
        return out


# 主网络：编码-瓶颈-解码结构
class MITM(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4]):
        super(MITM, self).__init__()
        self.dim = dim
        self.level = level

        self.embedding = nn.Conv2d(in_dim, dim, 3, 1, 1, bias=False)

        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                LGAB(dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),  # 特征下采样
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)   # 光照特征下采样
            ]))
            dim_level *= 2

        self.bottleneck = LGAB(dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1])

        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2, kernel_size=2),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                LGAB(dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim,
                     heads=(dim_level // 2) // dim),
            ]))
            dim_level //= 2

        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea):
        fea = self.embedding(x)
        fea_encoder = []
        illu_fea_list = []

        for (IGAB, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            fea = IGAB(fea, illu_fea)
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        fea = self.bottleneck(fea, illu_fea)

        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level - 1 - i]
            fea = LeWinBlcok(fea, illu_fea)

        out = self.mapping(fea) + x  # 残差连接
        return out


# 单阶段 HDRFormer 模块
class HDRFormer_Single_Stage(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2, num_blocks=[1, 1, 1]):
        super(HDRFormer_Single_Stage, self).__init__()
        self.estimator = Luminance_Estimator(n_feat)
        self.mitm = MITM(in_dim=in_channels,
                                 out_dim=out_channels,
                                 dim=n_feat,
                                 level=level,
                                 num_blocks=num_blocks)
        self.chroma = ChromaMLP(dim=3)

    def forward(self, img):
        luma_fea, luma_map = self.estimator(img)
        input_img = img * luma_map + img
        output_img = self.mitm(input_img, luma_fea)
        output_img = self.chroma(output_img)
        
        return output_img


# HDRFormer 主结构：多阶段堆叠
@ARCH_REGISTRY.register()
class HDRFormer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, stage=3, num_blocks=[1, 1, 1]):
        super(HDRFormer, self).__init__()
        self.stage = stage

        modules_body = [HDRFormer_Single_Stage(in_channels=in_channels,
                                               out_channels=out_channels,
                                               n_feat=n_feat,
                                               level=2,
                                               num_blocks=num_blocks)
                        for _ in range(stage)]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        return self.body(x)



if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis
    model = HDRFormer(stage=1,n_feat=40,num_blocks=[1,2,2]).cuda()
    print(model)
    inputs = torch.randn((1, 3, 512, 512)).cuda()
    output = model(inputs)
    print(f'output shape:{output.shape}')
    flops = FlopCountAnalysis(model,inputs)
    n_param = sum([p.nelement() for p in model.parameters()])  # 所有参数数量
    print(f'GMac:{flops.total()/(1024*1024*1024)}')
    print(f'Params:{n_param}')