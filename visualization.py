import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as T
from models_convnextv2 import convnextv2_large
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # 注册前向/反向传播钩子
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output.detach()

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, target_class=None):
        # 前向传播
        output = self.model(input_image)

        if target_class is None:
            target_class = torch.argmax(output, dim=1)

        # 反向传播获取梯度
        self.model.zero_grad()
        output[:, target_class].sum().backward(retain_graph=True)

        # 计算权重
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # 生成热力图
        cam = torch.sum(pooled_gradients * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # 只保留正相关区域
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam.squeeze().cpu().numpy()


def batch_process(root_dir, model, target_layer):
    """批量处理所有符合条件的图像"""
    # 图像预处理流水线
    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 获取所有待处理路径
    todo_list = get_valid_image_paths(root_dir)
    print(f"发现 {len(todo_list)} 张待处理图像")

    # 批量处理
    for idx, img_path in enumerate(todo_list, 1):
        try:
            process_image(img_path, transform, model, target_layer)
            print(f"进度：{idx}/{len(todo_list)} - 已处理：{img_path}")
        except Exception as e:
            print(f"处理失败：{img_path} - 错误：{str(e)}")


def get_valid_image_paths(root_dir):
    """获取所有需要处理的原始图像路径（排除已生成的CAM图像）"""
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_paths = []

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(subdir, file)
            base_name, ext = os.path.splitext(file)

            # 双重过滤：扩展名检查 + 排除已处理文件
            if (ext.lower() in valid_exts) and ('_cam' not in base_name):
                image_paths.append(file_path)

    return image_paths


def process_image(img_path, transform, model, target_layer):
    """处理单张图像并保存结果"""
    # 加载原始图像
    img = Image.open(img_path).convert('RGB')

    # 生成热力图
    input_tensor = transform(img).unsqueeze(0)
    cam = GradCAM(model, target_layer)
    cam_img = cam.generate_cam(input_tensor)

    img = cv2.cvtColor(np.array(img.resize((224, 224))), cv2.COLOR_RGB2BGR)
    heatmap = cv2.resize(cam_img, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    plt.figure(figsize=(10, 10),dpi=300)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    base, ext = os.path.splitext(img_path)
    save_path = f"{base}_cam{ext}"

    # 保存结果（保持原始格式）
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == '__main__':
    # 初始化模型
    # model = convnextv2_large(num_classes=5,pretrained=False)
    # pretrained_weights = torch.load("ConvNeXtV2_large_224_5_class_FOCAL/checkpoint-best.pth", map_location='cpu',weights_only=False)
    # model.load_state_dict(pretrained_weights['model'], strict=False)
    # model.eval()
    #
    # # 设置目标层
    # target_layer = model.stages[3][-1]
    #
    # root_directory = "example_image/ConvNeXtV2_large_5_class_FOCAL"
    #
    # batch_process(root_directory, model, target_layer)
    #
    #
    #
    #
    # model = convnextv2_large(num_classes=5,pretrained=False)
    # pretrained_weights = torch.load("ConvNeXtV2_large_224_5_class_CE/checkpoint-best.pth", map_location='cpu',weights_only=False)
    # model.load_state_dict(pretrained_weights['model'], strict=False)
    # model.eval()
    #
    # # 设置目标层
    # target_layer = model.stages[3][-1]
    #
    # root_directory = "example_image/ConvNeXtV2_large_5_class_CE"
    #
    # batch_process(root_directory, model, target_layer)
    #
    #
    #
    #
    # model = convnextv2_large(num_classes=5,pretrained=False)
    # pretrained_weights = torch.load("ConvNeXtV2_large_224_5_class_DICE/checkpoint-best.pth", map_location='cpu',weights_only=False)
    # model.load_state_dict(pretrained_weights['model'], strict=False)
    # model.eval()
    #
    # # 设置目标层
    # target_layer = model.stages[3][-1]
    #
    # root_directory = "example_image/ConvNeXtV2_large_5_class_DICE"
    #
    # batch_process(root_directory, model, target_layer)
    #
    #
    #
    # model = convnextv2_large(num_classes=5,pretrained=False)
    # pretrained_weights = torch.load("ConvNeXtV2_large_224_5_class_WCE/checkpoint-best.pth", map_location='cpu',weights_only=False)
    # model.load_state_dict(pretrained_weights['model'], strict=False)
    # model.eval()
    #
    # # 设置目标层
    # target_layer = model.stages[3][-1]
    #
    # root_directory = "example_image/ConvNeXtV2_large_5_class_WCE"
    #
    # batch_process(root_directory, model, target_layer)
    #
    #
    #
    # model = convnextv2_large(num_classes=5,pretrained=False)
    # pretrained_weights = torch.load("ConvNeXtV2_large_224_5_class_TVERSKY/checkpoint-best.pth", map_location='cpu',weights_only=False)
    # model.load_state_dict(pretrained_weights['model'], strict=False)
    # model.eval()
    #
    # # 设置目标层
    # target_layer = model.stages[3][-1]
    #
    # root_directory = "example_image/ConvNeXtV2_large_5_class_TVERSKY"
    #
    # batch_process(root_directory, model, target_layer)


    model = convnextv2_large(num_classes=5,pretrained=False)
    pretrained_weights = torch.load("ConvNeXtV2_large_224_5_class_GHM/checkpoint-best.pth", map_location='cpu',weights_only=False)
    model.load_state_dict(pretrained_weights['model'], strict=False)
    model.eval()

    # 设置目标层
    target_layer = model.stages[3][-1]

    root_directory = "example_image/ConvNeXtV2_large_5_class_GHM"

    batch_process(root_directory, model, target_layer)