import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import os
import os
from pathlib import Path


def count_current_folder_images(folder_path: str) -> int:

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}

    # 获取文件夹内所有条目
    all_entries = os.listdir(folder_path)

    count = 0
    for entry in all_entries:
        # 拼接完整路径
        full_path = os.path.join(folder_path, entry)
        # 仅统计文件（排除文件夹）
        if os.path.isfile(full_path):
            ext = Path(entry).suffix.lower()
            if ext in image_extensions:
                count += 1
    return count

def get_sample_num_under_classes(root, cls_task):
    '''
            0 : ["anodr"],
            1 : ["bmilddr"],
            2 : ["cmoderatedr"],
            3 : ["dseveredr"],
            4 : ["eproliferativedr"]
    '''
    num_per_classes = [0, 0, 0, 0, 0]
    folder_names = ['anodr', 'bmilddr', 'cmoderatedr', 'dseveredr', 'eproliferativedr']
    index = 0
    for folder in folder_names:
        folder_path = os.path.join(root, folder)
        num_per_classes[index] = count_current_folder_images(folder_path)
        index = index + 1

    if cls_task == 5:
        return num_per_classes
    elif cls_task == 3:
        num_per_classes = [num_per_classes[0],
                           num_per_classes[1] + num_per_classes[2] + num_per_classes[3],
                           num_per_classes[4]]
        return num_per_classes
    elif cls_task == 2:
        num_per_classes = [num_per_classes[0],
                           num_per_classes[1] + num_per_classes[2] + num_per_classes[3] + num_per_classes[4]]
        return num_per_classes
    else:
        raise NotImplementedError

def init_loss(args):
    # get the class weight:
    train_root = os.path.join(args.data_path, 'train')
    weights = get_sample_num_under_classes(train_root, args.cls_task)
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = 1.0 / torch.sqrt(weights)  # sqrt compress
    weights = weights / weights.sum()

    if args.loss_func == 'CE':
        # a standard cross entropy loss
        return torch.nn.CrossEntropyLoss().cuda()
    elif args.loss_func == 'WCE':
        print('the weights for WCE at each class is {}'.format(weights))
        return torch.nn.CrossEntropyLoss(weight=weights).cuda()

    elif args.loss_func == 'FOCAL':
        print('the weights for multi class Focal loss at each class is {}'.format(weights))
        return focal_loss(gamma=2, alpha=weights).cuda()

    elif args.loss_func == 'GHM':
        if args.cls_task ==2:
            bins = 10
            mom = 0.75
        elif args.cls_task ==3:
            bins =10
            mom = 0.7
        elif args.cls_task ==5:
            bins = 15
            mom = 0.65
        else:
            raise NotImplementedError
        return GHMC_Loss(bins=bins, alpha=mom, num_classes=args.cls_task,
                         task_type='binary' if args.cls_task == 2 else 'multiclass').cuda()

    elif args.loss_func == 'DICE':

        if args.cls_task ==2:
            return DiceLoss(mode = 'binary-softmax').cuda()
        elif args.cls_task ==3 or args.cls_task ==5:
            return DiceLoss(mode='multiclass').cuda()
        else:
            raise NotImplementedError

    elif args.loss_func == 'TVERSKY':
        if args.cls_task ==2:
            return TverskyLoss(mode = 'binary-softmax').cuda()
        elif args.cls_task ==3 or args.cls_task ==5:
            return TverskyLoss(mode='multiclass').cuda()
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError




class TverskyLoss(nn.Module):
    def __init__(self,
                 alpha: float = 0.4,
                 beta: float = 0.6,
                 smooth: float = 1e-6,
                 mode: str = 'auto',  # 'auto', 'binary-sigmoid', 'binary-softmax', 'multiclass'
                 class_weights: torch.Tensor = None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.mode = mode
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 自动检测模式
        if self.mode == 'auto':
            if logits.size(1) == 1:
                self.mode = 'binary-sigmoid'
            elif logits.size(1) == 2:
                self.mode = 'binary-softmax'
            else:
                self.mode = 'multiclass'

        # 二分类 Sigmoid 模式
        if self.mode == 'binary-sigmoid':
            probs = torch.sigmoid(logits)  # [B, 1]
            targets = targets.view(-1, 1).float()  # [B, 1]

            tp = (probs * targets).sum()  # True Positives
            fp = (probs * (1 - targets)).sum()  # False Positives
            fn = ((1 - probs) * targets).sum()  # False Negatives

            numerator = tp
            denominator = tp + self.alpha * fp + self.beta * fn + self.smooth
            loss = 1 - (numerator / denominator)

        # 二分类 Softmax 或多分类模式
        else:
            num_classes = logits.size(1)
            probs = F.softmax(logits, dim=1)  # [B, C]
            targets_onehot = F.one_hot(targets, num_classes).float()  # [B, C]
            targets_onehot = targets_onehot.to(probs.device)  # 确保设备一致

            tp = (probs * targets_onehot).sum(dim=0)  # [C]
            fp = (probs * (1 - targets_onehot)).sum(dim=0)  # [C]
            fn = ((1 - probs) * targets_onehot).sum(dim=0)  # [C]

            numerator = tp
            denominator = tp + self.alpha * fp + self.beta * fn + self.smooth
            tversky_per_class = numerator / denominator  # [C]

            if self.class_weights is not None:
                tversky_per_class *= self.class_weights.to(probs.device)

            loss = 1 - tversky_per_class.mean()

        return loss


class DiceLoss(nn.Module):
    def __init__(self,
                 mode: str = 'auto',  # 'auto', 'binary-sigmoid', 'binary-softmax', 'multiclass'
                 smooth: float = 1e-6,
                 class_weights: torch.Tensor = None,
                 ignore_index: int = -100):

        super().__init__()
        self.smooth = smooth
        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.mode = mode

        if self.mode not in ['auto', 'binary-sigmoid', 'binary-softmax', 'multiclass']:
            raise ValueError(f"Invalid mode: {self.mode}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.mode == 'auto':
            if logits.size(1) == 1:
                self.mode = 'binary-sigmoid'
            elif logits.size(1) == 2:
                self.mode = 'binary-softmax'
            else:
                self.mode = 'multiclass'
        mask = targets != self.ignore_index
        logits = logits[mask]
        targets = targets[mask]
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)

        if self.mode == 'binary-sigmoid':
            probs = torch.sigmoid(logits)  # [B, 1]
            targets = targets.view(-1, 1).float()  # [B, 1]
            intersection = (probs * targets).sum()
            union = probs.sum() + targets.sum()
            dice = (2. * intersection + self.smooth) / (union + self.smooth)

        else:
            num_classes = logits.size(1)
            probs = F.softmax(logits, dim=1)  # [B, C]
            targets_onehot = F.one_hot(targets, num_classes).float()  # [B, C]
            targets_onehot = targets_onehot.to(probs.device)

            intersection = (probs * targets_onehot).sum(dim=0)  # [C]
            union = probs.sum(dim=0) + targets_onehot.sum(dim=0)  # [C]

            dice_per_class = (2. * intersection + self.smooth) / (union + self.smooth)
            if self.class_weights is not None:
                dice_per_class *= self.class_weights.to(probs.device)
            dice = dice_per_class.mean()

        return 1 - dice


class UCircle(nn.Module):

    def __init__(self, margin = 0.35, gamma = 128):
        super(UCircle, self).__init__()
        self.margin = margin
        self.gamma = gamma

    def forward(self, embedding: torch.Tensor, targets: torch.Tensor):
        embedding = F.normalize(embedding, dim=1)
        dist_mat = torch.matmul(embedding, embedding.t())

        N = dist_mat.size(0)

        is_pos_1 = targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
        is_pos_1 = is_pos_1 * (targets.view(N, 1).expand(N, N).eq(1).float())

        is_neg = targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()

        # Mask scores related to itself
        is_pos_1 = is_pos_1 - torch.eye(N, N, device=is_pos_1.device)

        s_p_1 = dist_mat * is_pos_1
        s_n = dist_mat * is_neg

        alpha_p_1 = torch.clamp_min(-s_p_1.detach() + 1 + self.margin, min=0.)
        alpha_n = torch.clamp_min(s_n.detach() + self.margin, min=0.)
        delta_p = 1 - self.margin
        delta_n = self.margin

        logit_p_1 = - self.gamma * alpha_p_1 * (s_p_1 - delta_p) + (-99999999.) * (1 - is_pos_1)
        logit_n = self.gamma * alpha_n * (s_n - delta_n) + (-99999999.) * (1 - is_neg)

        loss = F.softplus(torch.logsumexp(logit_p_1, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()

        loss = loss/N

        return loss


class GHM_Loss(nn.Module):
    def __init__(self, bins=10, alpha=0.5):
        super(GHM_Loss, self).__init__()
        self._bins = bins
        self._alpha = alpha
        self.register_buffer('_last_bin_count', None)

    def _g2bin(self, g):
        return torch.floor(g * (self._bins - 0.0001)).long()

    def _custom_loss(self, x, target, weight):
        raise NotImplementedError

    def _custom_loss_grad(self, x, target):
        raise NotImplementedError

    def forward(self, x, target):
        device = x.device
        g = self._custom_loss_grad(x, target).detach()

        bin_idx = self._g2bin(g)

        bin_count = torch.zeros(self._bins, device=device)
        for i in range(self._bins):
            bin_count[i] = (bin_idx == i).sum().item()

        N = x.size(0)  # 样本数

        if self._last_bin_count is None:
            self._last_bin_count = bin_count
        else:
            bin_count = self._alpha * self._last_bin_count + (1 - self._alpha) * bin_count
            self._last_bin_count = bin_count.clone()

        nonempty_bins = (bin_count > 0).sum().item()
        gd = bin_count * nonempty_bins
        gd = torch.clamp(gd, min=1e-6)
        beta = N / gd

        sample_weights = beta[bin_idx]
        return self._custom_loss(x, target, sample_weights)


class GHMC_Loss(GHM_Loss):
    def __init__(self, bins=10, alpha=0.5, task_type='binary', num_classes=1):

        super().__init__(bins, alpha)
        self.task_type = task_type
        self.num_classes = num_classes

        if self.task_type not in ['binary', 'multiclass']:
            raise ValueError("task_type must be 'binary' or 'multiclass'")

        if self.task_type == 'binary' and num_classes not in [1, 2]:
            raise ValueError("For binary task, num_classes must be 1 (Sigmoid) or 2 (Softmax)")

    def _custom_loss_grad(self, x, target):
        if self.task_type == 'binary':
            if self.num_classes == 1:

                probs = torch.sigmoid(x).squeeze(1).detach()  # [batch_size]
                grad = torch.abs(probs - target.float())  # [batch_size]
            else:
                probs = F.softmax(x, dim=1).detach()  # [batch_size, 2]
                target_onehot = F.one_hot(target, num_classes=2).float().to(x.device)
                grad = torch.abs(probs - target_onehot).mean(dim=1)  # [batch_size]
        else:
            probs = F.softmax(x, dim=1).detach()
            target_onehot = F.one_hot(target, num_classes=self.num_classes).float().to(x.device)
            grad = torch.abs(probs - target_onehot).mean(dim=1)
        return grad

    def _custom_loss(self, x, target, sample_weights):
        if self.task_type == 'binary':
            if self.num_classes == 1:
                # 单节点BCEWithLogitsLoss
                ce_loss = F.binary_cross_entropy_with_logits(
                    x.squeeze(1), target.float(), reduction='none'
                )
            else:
                # 双节点CrossEntropyLoss
                ce_loss = F.cross_entropy(x, target, reduction='none')
        else:
            # 多分类CrossEntropy
            ce_loss = F.cross_entropy(x, target, reduction='none')

        return (ce_loss * sample_weights).mean()


class BinaryFocalLoss(nn.Module):

    def __init__(self, alpha=1, gamma=2, reduction='mean', **kwargs):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

    def forward(self, output, target):
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        target = target.unsqueeze(dim=1)
        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -pos_weight * torch.log(prob)  # / (torch.sum(pos_weight) + 1e-4)

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -self.alpha * neg_weight * F.logsigmoid(-output)  # / (torch.sum(neg_weight) + 1e-4)

        loss = pos_loss + neg_loss
        loss = loss.mean()
        return loss

class FocalLoss(nn.Module):

    def __init__(self,
                 alpha = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):

        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


def focal_loss(alpha = None,
               gamma: float = 0.,
               reduction: str = 'mean',
               ignore_index: int = -100,
               device='cpu',
               dtype=torch.float32) -> FocalLoss:

    if alpha is not None:
        if not isinstance(alpha, Tensor):
            alpha = torch.tensor(alpha)
        alpha = alpha.to(device=device, dtype=dtype)

    fl = FocalLoss(
        alpha=alpha,
        gamma=gamma,
        reduction=reduction,
        ignore_index=ignore_index)
    return fl


class CDW_CELoss(nn.Module):

    def __init__(self, num_classes: int,
                 alpha: float = 2.,
                 delta: float = 3.,
                 reduction: str = "mean",
                 transform: str = "power",  # Original paper uses power transform
                 eps: float = 1e-8):
        super(CDW_CELoss, self).__init__()

        assert alpha > 0, "Alpha should be larger than 0"
        assert reduction in [
            "mean", "sum"], "Reduction should be either mean or sum"
        assert transform in [
            "huber", "log", "power"], "Transform should be either huber, log or power"

        self.reduction = reduction
        self.transform = transform
        self.alpha = alpha
        self.eps = eps
        self.num_classes = num_classes
        self.register_buffer(name="w", tensor=torch.tensor(
            [float(i) for i in range(self.num_classes)]).cuda())  # to speed up the computation

        self.delta = delta  # for huber transform only

    def huber_transform(self, x):
        """Weight distances according to the Huber Loss"""
        return torch.where(
            x < self.delta,
            0.5 * torch.pow(x, 2),
            self.delta * (x - 0.5 * self.delta)
        )

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:

        w = torch.abs(self.w - target.view(-1, 1))  # calculate penalty weights

        if self.transform == "huber":
            # apply huber transform (not in the paper)
            w = self.huber_transform(w)
        elif self.transform == "log":
            w = torch.log1p(w)
            # apply log transform (not in the paper)
            w = torch.pow(w, self.alpha)
        elif self.transform == "power":
            # apply power transform (in the paper)
            w = torch.pow(w, self.alpha)
        else:
            raise NotImplementedError(
                "%s transform is not implemented" % self.transform)

        loss = - torch.mul(torch.log(1 - logits + self.eps), w).sum(-1)

        if self.reduction == "mean":
            return torch.mean(loss)
        elif self.reduction == "sum":
            return torch.sum(loss)
        else:
            raise NotImplementedError(
                "%s reduction is not implemented" % self.reduction)


if __name__ == '__main__':
    nums = get_sample_num_under_classes('..\\..\\APTOS2019\\train',cls_task=5)
    nums = torch.tensor(nums, dtype=torch.float32)
    print(nums)
    nums = 1.0 / nums
    nums = nums / nums.sum()
    print(nums)