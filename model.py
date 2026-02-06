import torch
import torch.nn as nn
import math

__all__ = ['mobilenetv3_large', 'mobilenetv3_small']#define the public API of this model

#funtion to ensure all the lays have a channel number that is divisible by 8 to adapt to hardware and improve inference speed
def _make_divisible(v, divisor, min_value=None):#min_value:set divisor to min_value if not provided
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v+int(divisor/2) //divisor*divisor))
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

#Define the h_sigmoid and h_swish activation function
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace)

    def forward(self, x):
        return self.relu(x+3)/6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace)

    def forward(self, x):
        return x*self.sigmoid(x)


#define the model of SE
class SElayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SElayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#Global average pooling to compress spatial dimensions to 1x1 (Squeeze)
        self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction,8)),
                nn.ReLU(inplace=True),
                nn.Linear( _make_divisible(channel // reduction,8),channel),
                h_sigmoid()


        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )

class invertedResidual(nn.Module):
    def __init__(self, inp,hidden_dim, oup, kernel_size,stride, use_se, use_hs):
        super(invertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity=(stride == 1 and inp == oup)

        if inp==hidden_dim:
            self.conv = nn.Sequential(
                #dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, bias=False),#(kernel_size - 1) // 2=padding
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                #SE
                SElayer(hidden_dim) if use_se else nn.Identity(),
                #pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),

            )
        else:
            self.conv = nn.Sequential(
                #pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                #dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, bias=False),
                # (kernel_size - 1) // 2=padding
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # SE
                SElayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class mobilenetv3(nn.Module):
    def __init__(self, cfgs, mode, num_classes=1000, width_mult=1.):
        super(mobilenetv3, self).__init__()
        self.cfgs = cfgs
        assert mode in ["large", "small"]
        #build first layer
        input_channel =_make_divisible(16* width_mult, 8)
        layers=[conv_3x3_bn(3, input_channel, 2)]
        #build inverted residual blocks
        block=invertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            exp_size=_make_divisible(input_channel* width_mult, 8)
            output_channel = _make_divisible(c * width_mult, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel=output_channel
        self.features = nn.Sequential(*layers)
        #build last several layers
        self.conv=conv_1x1_bn(input_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        output_channel = {'large': 1280, 'small': 1024}
        output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[
            mode]
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x =torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenetv3_large(**kwargs):
    cfgs = [
        # k, t, c, SE, HS, s
        [3, 1, 16, 0, 0, 1],
        [3, 4, 24, 0, 0, 2],
        [3, 3, 24, 0, 0, 1],
        [5, 3, 40, 1, 0, 2],
        [5, 3, 40, 1, 0, 1],
        [5, 3, 40, 1, 0, 1],
        [3, 6, 80, 0, 1, 2],
        [3, 2.5, 80, 0, 1, 1],
        [3, 2.3, 80, 0, 1, 1],
        [3, 2.3, 80, 0, 1, 1],
        [3, 6, 112, 1, 1, 1],
        [3, 6, 112, 1, 1, 1],
        [5, 6, 160, 1, 1, 2],
        [5, 6, 160, 1, 1, 1],
        [5, 6, 160, 1, 1, 1]
    ]

    return mobilenetv3(cfgs,mode='large', **kwargs)

def mobilenetv3_small(**kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3,    1,  16, 1, 0, 2],
        [3,  4.5,  24, 0, 0, 2],
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
        [5,    6,  40, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    6,  96, 1, 1, 2],
        [5,    6,  96, 1, 1, 1],
        [5,    6,  96, 1, 1, 1],
    ]

    return MobileNetV3(cfgs, mode='small', **kwargs)

#PyTorch 中，只有继承nn.Module的组件，才能被注册为模型的一部分，参与前向传播、反向求导、设备迁移（CPU/GPU），所以定义h_sigmoid的时候使用类而不是函数
# 普通函数无法被model.to(device)识别，无法和模型一起迁移到 GPU
#view()：PyTorch 中用于重塑张量形状的函数，不改变数据本身，只调整维度结构
#nn.Linear 只能处理二维张量，格式固定为 [批次大小, 特征维度],.view(b, c) 会把张量的维度重新组合，剔除末尾两个无意义的1
#hidden_dim逆残差块中间隐藏层的通道数，也叫扩展通道数 / 升维通道数，是 MobileNet 逆残差结构的核心参数，作用：让轻量化的 DW 卷积在高维空间提取更丰富的特征，弥补 DW 卷积仅提取空间特征、特征表达弱的缺陷。
#倒残差类里面self.identity = stride == 1 and inp == oup这一句只是为第105行定义了布尔值，如果为true就执行下一行，并不影响这个类是否运行，类运行的条件是实例化+赋值
#hidden_dim是指模块内部中间层的特征通道数，逆残差第一步就是将inp提升到hidden_dim，这是dw卷积的前置条件，只有提升到hidden_dim才能进行dw操作
#cfgs 网络配置表:提前写好的参数列表，规定每一层逆残差块的卷积核、步长、通道数等;num_classes:分类类别数 默认1000类，可以修改为自己需要的如猫狗二分类；width_mult 宽度缩放因子 缩放通道数，调整模型大小和计算量；exp_size：逆残差块的hidden_dim（中间工作通道）
#115行的基础通道选16是因为在「足够的特征表达能力」和「最小的计算量」之间找到平衡，是移动端网络的经典选择。
#下采样：通过特定操作，降低图像 / 特征图的「空间分辨率」（宽 × 高），减少像素点数量，同时尽可能保留核心信息。
#Python 函数传参的核心规则 ——按位置传参时，只看顺序，不看变量本身的名字。解释了68行和122行参数名称不同但是还是能正确传参的原因
#exp_size在逆残差块中是作为中间隐藏通道存在，并且它的值大于其中的output_channel，但是在最后几层时是输出通道。
#池化层的核心作用是压缩空间维度（高、宽），不会改变通道维度和批次维度，nn.AdaptiveAvgPool2d((1, 1))比如这个函数它的意思是压缩图像长宽使其为1,1
#第129行的out_channel是分类器第一层全连接层的神经元个数，大模型用 1280 个神经元，小模型用 1024 个神经元，是人为设计的超参数。（处理后输出多少个数据）
#assert mode in ["large", "small"] output_channel = {'large': 1280, 'small': 1024} output_channel[mode]  结合这三行代码，output_channel输出由mode进行索引的数字
#nn.Dropout(0.2) = 训练时随机屏蔽 20% 的神经元，是防止模型过拟合的工具；
#self._initialize_weights()虽然没有forward中使用，但是模型初始化阶段的固定操作，在创建模型实例的时候会自动运行，函数名以下划线 _ 开头（_initialize_weights），是 Python 的约定：代表这是类内部私有方法，只在类内部使用，不对外暴露调用，所以你不会在外部代码看到手动调用它的场景。
#self.modules() 是 PyTorch 内置方法，会递归获取模型里所有的网络层和子模块，比如卷积、BN、全连接、逆残差块等，逐个处理。\
#x = torch.flatten(x, 1)代表从第一个维度开始倒最后一个维度结束合并中间所有维度，保留第0维度不变；x = x.view(x.size(0), -1) 手动指定第 0 维保持原大小，剩余所有元素自动合并为第 1 维。
#x.size(0)：获取张量第 0 个维度的长度
#在 Python 函数 / 类构造方法中，参数是否为必选，完全由书写格式决定,参数后面没有赋值（=默认值），就是必选参数；有赋值，就是可选参数