import torch
import torch.nn as nn
import numpy as np

from utils.misc import multiclass_nms

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def _adapt_first_conv_weight(state_dict, in_channels):
    """将 ImageNet 预训练的 3 通道 conv1 权重适配为 in_channels."""
    key = 'conv1.weight'
    if key not in state_dict:
        return state_dict
    w = state_dict[key]          # [64,3,7,7]
    if in_channels == 3:
        return state_dict
    elif in_channels == 1:
        # 取均值 -> [64,1,7,7]
        w_new = w.mean(dim=1, keepdim=True)
    elif in_channels == 2:
        # 取前两通道或均值复制 (这里用: 前两通道；若原 w 只有3通道足够)
        w_new = w[:, :2, :, :].clone()
    else:
        # >3: 随机追加剩余通道 (保持尺度)
        extra = in_channels - 3
        rand_extra = torch.randn(w.size(0), extra, w.size(2), w.size(3)) * w.std()
        w_new = torch.cat([w, rand_extra], dim=1)
    state_dict[key] = w_new
    return state_dict

def get_conv2d(c1, c2, k, p, s, d, g, bias=False):
    conv = nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=bias)

    return conv

def get_activation(act_type=None):
    if act_type == 'relu':
        return nn.ReLU(inplace=True)
    elif act_type == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif act_type == 'mish':
        return nn.Mish(inplace=True)
    elif act_type == 'silu':
        return nn.SiLU(inplace=True)
    elif act_type is not None:
        return nn.Identity()
    else:
        raise NotImplementedError('Activation {} not implemented.'.format(act_type))

def get_norm(norm_type, dim):
    if norm_type == 'BN':
        return nn.BatchNorm2d(dim)
    elif norm_type == 'GN':
        return nn.GroupNorm(num_groups=32, num_channels=dim)
    elif norm_type is not None:
        return nn.Identity()
    else:
        raise NotImplementedError('Normalization {} not implemented.'.format(norm_type))

class Conv(nn.Module):
    def __init__(self, 
                 c1,                   # 输入通道数
                 c2,                   # 输出通道数 
                 k=1,                  # 卷积核尺寸 
                 p=0,                  # 补零的尺寸
                 s=1,                  # 卷积的步长
                 d=1,                  # 卷积膨胀系数
                 act_type='lrelu',     # 激活函数的类别
                 norm_type='BN',       # 归一化层的类别
                 depthwise=False       # 是否使用depthwise卷积
                 ):
        super(Conv, self).__init__()
        convs = []
        add_bias = False if norm_type else True

        # 构建depthwise + pointwise卷积
        if depthwise:
            convs.append(get_conv2d(c1, c1, k=k, p=p, s=s, d=d, g=c1, bias=add_bias))
            # 首先，搭建depthwise卷积
            if norm_type:
                convs.append(get_norm(norm_type, c1))
            if act_type:
                convs.append(get_activation(act_type))
            # 然后，搭建pointwise卷积
            convs.append(get_conv2d(c1, c2, k=1, p=0, s=1, d=d, g=1, bias=add_bias))
            if norm_type:
                convs.append(get_norm(norm_type, c2))
            if act_type:
                convs.append(get_activation(act_type))

        # 构建普通的标准卷积
        else:
            convs.append(get_conv2d(c1, c2, k=k, p=p, s=s, d=d, g=1, bias=add_bias))
            if norm_type:
                convs.append(get_norm(norm_type, c2))
            if act_type:
                convs.append(get_activation(act_type))
            
        self.convs = nn.Sequential(*convs)


    def forward(self, x):
        return self.convs(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False, in_channels=3):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Input:
            x: (Tensor) -> [B, C, H, W]
        Output:
            c5: (Tensor) -> [B, C, H/32, W/32]
        """
        c1 = self.conv1(x)     # [B, C, H/2, W/2]
        c1 = self.bn1(c1)      # [B, C, H/2, W/2]
        c1 = self.relu(c1)     # [B, C, H/2, W/2]
        c2 = self.maxpool(c1)  # [B, C, H/4, W/4]

        c2 = self.layer1(c2)   # [B, C, H/4, W/4]
        c3 = self.layer2(c2)   # [B, C, H/8, W/8]
        c4 = self.layer3(c3)   # [B, C, H/16, W/16]
        c5 = self.layer4(c4)   # [B, C, H/32, W/32]

        return c5

# Spatial Pyramid Pooling
class SPPF(nn.Module):
    """
        该代码参考YOLOv5的官方代码实现 https://github.com/ultralytics/yolov5
    """
    def __init__(self, in_dim, out_dim, expand_ratio=0.5, pooling_size=5, act_type='lrelu', norm_type='BN'):
        super().__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.out_dim = out_dim
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = Conv(inter_dim * 4, out_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.m = nn.MaxPool2d(kernel_size=pooling_size, stride=1, padding=pooling_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)

        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

# 解偶检测头
class DecoupledHead(nn.Module):
    def __init__(self, cfg, in_dim, out_dim, num_classes=80):
        super().__init__()
        print('==============================')
        print('Head: Decoupled Head')
        self.in_dim = in_dim
        self.num_cls_head=cfg['num_cls_head']
        self.num_reg_head=cfg['num_reg_head']
        self.act_type=cfg['head_act']
        self.norm_type=cfg['head_norm']

        # ------------------ 类别检测头 ------------------
        cls_feats = []
        self.cls_out_dim = max(out_dim, num_classes)
        for i in range(cfg['num_cls_head']):
            if i == 0:
                cls_feats.append(
                    Conv(in_dim, self.cls_out_dim, k=3, p=1, s=1, 
                        act_type=self.act_type,
                        norm_type=self.norm_type,
                        depthwise=cfg['head_depthwise'])
                        )
            else:
                cls_feats.append(
                    Conv(self.cls_out_dim, self.cls_out_dim, k=3, p=1, s=1, 
                        act_type=self.act_type,
                        norm_type=self.norm_type,
                        depthwise=cfg['head_depthwise'])
                        )
                
        # ------------------ 回归检测头 ------------------
        reg_feats = []
        self.reg_out_dim = max(out_dim, 64)
        for i in range(cfg['num_reg_head']):
            if i == 0:
                reg_feats.append(
                    Conv(in_dim, self.reg_out_dim, k=3, p=1, s=1, 
                        act_type=self.act_type,
                        norm_type=self.norm_type,
                        depthwise=cfg['head_depthwise'])
                        )
            else:
                reg_feats.append(
                    Conv(self.reg_out_dim, self.reg_out_dim, k=3, p=1, s=1, 
                        act_type=self.act_type,
                        norm_type=self.norm_type,
                        depthwise=cfg['head_depthwise'])
                        )

        self.cls_feats = nn.Sequential(*cls_feats)
        self.reg_feats = nn.Sequential(*reg_feats)

    def forward(self, x):
        """
            x: (torch.Tensor) [B, C, H, W]
        """
        cls_feats = self.cls_feats(x)
        reg_feats = self.reg_feats(x)

        return cls_feats, reg_feats

# YOLOv1
class YOLORTv1(nn.Module):
    def __init__(self,
                 cfg,
                 device,
                 img_size=None,
                 num_classes=3,
                 conf_thresh=0.01,
                 nms_thresh=0.5,
                 trainable=False,
                 deploy=False):
        super(YOLORTv1, self).__init__()
        # ------------------- 基础参数 -------------------
        self.cfg = cfg                     # 模型配置文件
        self.img_size = img_size           # 输入图像大小
        self.device = device               # cuda或者是cpu
        self.num_classes = num_classes     # 类别的数量
        self.trainable = trainable         # 训练的标记
        self.conf_thresh = conf_thresh     # 得分阈值
        self.nms_thresh = nms_thresh       # NMS阈值
        self.stride = 32                   # 网络的最大步长
        self.deploy = deploy
        
        # ------------------- 网络结构 -------------------
        ## 主干网络
        in_channels = cfg.get('in_channels', 1)
        self.backbone = ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels)
        if cfg.get('pretrained', True):
            state_dict = torch.load("./weights/resnet18-5c106cde.pth",
                                    map_location=lambda storage, loc: storage)
            state_dict = _adapt_first_conv_weight(state_dict, in_channels)
            self.backbone.load_state_dict(state_dict, strict=False)
        feat_dim=512

        ## 颈部网络
        out_dim=512
        self.neck = SPPF(
            in_dim=feat_dim,
            out_dim=out_dim,
            expand_ratio=cfg['expand_ratio'], 
            pooling_size=cfg['pooling_size'],
            act_type=cfg['neck_act'],
            norm_type=cfg['neck_norm']
            )
        head_dim = self.neck.out_dim

        ## 检测头
        self.head = DecoupledHead(cfg, head_dim, head_dim, num_classes) 

        ## 预测层
        self.obj_pred = nn.Conv2d(head_dim, 1, kernel_size=1)
        self.cls_pred = nn.Conv2d(head_dim, num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(head_dim, 4, kernel_size=1)
        
        # -------------- 初始化预测层的参数 --------------
        # Init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        # obj pred
        b = self.obj_pred.bias.view(1, -1)
        b.data.fill_(bias_value.item())
        self.obj_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        # cls pred
        b = self.cls_pred.bias.view(1, -1)
        b.data.fill_(bias_value.item())
        self.cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        # reg pred
        b = self.reg_pred.bias.view(-1, )
        b.data.fill_(1.0)
        self.reg_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        w = self.reg_pred.weight
        w.data.fill_(0.)
        self.reg_pred.weight = torch.nn.Parameter(w, requires_grad=True)
    
    def create_grid(self, fmp_size):
        """ 
            用于生成G矩阵，其中每个元素都是特征图上的像素坐标。
        """
        # 特征图的宽和高
        ws, hs = fmp_size

        # 生成网格的x坐标和y坐标
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)],indexing='ij')

        # 将xy两部分的坐标拼起来：[H, W, 2]
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()

        # [H, W, 2] -> [HW, 2] -> [HW, 2]
        grid_xy = grid_xy.view(-1, 2).to(self.device)
        
        return grid_xy

    def decode_boxes(self, pred_reg, fmp_size):
        """
            将YOLO预测的 (tx, ty)、(tw, th) 转换为bbox的左上角坐标 (x1, y1) 和右下角坐标 (x2, y2)。
            输入:
                pred_reg: (torch.Tensor) -> [B, HxW, 4] or [HxW, 4]，网络预测的txtytwth
                fmp_size: (List[int, int])，包含输出特征图的宽度和高度两个参数
            输出:
                pred_box: (torch.Tensor) -> [B, HxW, 4] or [HxW, 4]，解算出的边界框坐标
        """
        # 生成网格坐标矩阵
        grid_cell = self.create_grid(fmp_size)

        # 计算预测边界框的中心点坐标和宽高
        pred_ctr = (torch.sigmoid(pred_reg[..., :2]) + grid_cell) * self.stride
        wh_log = torch.clamp(pred_reg[..., 2:], min=-10.0, max=10.0)
        pred_wh = torch.exp(wh_log) * self.stride

        # 将所有bbox的中心带你坐标和宽高换算成x1y1x2y2形式
        pred_x1y1 = pred_ctr - pred_wh * 0.5
        pred_x2y2 = pred_ctr + pred_wh * 0.5
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)
        pred_box[~torch.isfinite(pred_box)] = 0.

        return pred_box

    def postprocess(self, bboxes, obj_scores):
        """
            仅目标存在与否的后处理：
            输入:
                bboxes: (numpy.array) -> [HxW, 4]
                obj_scores: (numpy.array) -> [HxW,]
            输出:
                bboxes: (numpy.array) -> [N, 4]
                score:  (numpy.array) -> [N,]
                labels: (numpy.array) -> [N,]  全部为 0
        """
        # 阈值筛选
        keep = np.where(obj_scores >= self.conf_thresh)[0]
        bboxes = bboxes[keep]
        scores = obj_scores[keep]
        labels = np.zeros(len(scores), dtype=np.int32)

        # 进一步过滤非法框
        finite = np.isfinite(bboxes).all(axis=1)
        proper = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])
        keep2 = finite & proper
        bboxes = bboxes[keep2]
        scores = scores[keep2]
        labels = labels[keep2]

        # 类无关 NMS
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, 1, True)

        return bboxes, scores, labels

    @torch.no_grad()
    def inference(self, x):
        feat = self.backbone(x)
        feat = self.neck(feat)
        cls_feat, reg_feat = self.head(feat)

        # 仅计算 obj 与 bbox
        obj_pred = self.obj_pred(cls_feat)
        reg_pred = self.reg_pred(reg_feat)
        fmp_size = obj_pred.shape[-2:]

        obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)  # [B,HW,1]
        reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)  # [B,HW,4]

        # 测试默认 batch=1
        obj_pred = obj_pred[0].squeeze(-1)    # [HW,]
        reg_pred = reg_pred[0]                # [HW,4]

        bboxes = self.decode_boxes(reg_pred, fmp_size)

        # 分数仅用 obj
        obj_scores = obj_pred.sigmoid()

        if self.deploy:
            # 导出形状为 [n_anchors_all, 4+1]
            outputs = torch.cat([bboxes, obj_scores[:, None]], dim=-1)
            return outputs
        else:
            bboxes = bboxes.cpu().numpy()
            obj_scores = obj_scores.cpu().numpy()
            bboxes, scores, labels = self.postprocess(bboxes, obj_scores)
            return bboxes, scores, labels

    def forward(self, x):
        if not self.trainable:
            return self.inference(x)
        else:
            feat = self.backbone(x)
            feat = self.neck(feat)
            cls_feat, reg_feat = self.head(feat)

            obj_pred = self.obj_pred(cls_feat)
            reg_pred = self.reg_pred(reg_feat)
            fmp_size = obj_pred.shape[-2:]

            obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)   # [B, M, 1]
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().flatten(1, 2)   # [B, M, 4]

            box_pred = self.decode_boxes(reg_pred, fmp_size)

            outputs = {
                "pred_obj": obj_pred,   # [B, M, 1]
                "pred_box": box_pred,   # [B, M, 4]
                "stride": self.stride,
                "fmp_size": fmp_size
            }
            return outputs

