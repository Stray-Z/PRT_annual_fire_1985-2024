import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import testDataset
from UMNet import UMNet
from net.unet import UNet
# from net.unet2 import UNet
from net.fasternet import FasterNet
# 定义颜色映射
PALETTE = {
    0: [255, 255, 0],  # 黄色 - 火灾
    1: [0, 0, 255],  # 蓝色 - 损失
    255: [255, 255, 255]  # 白色 - 背景
}

# 路径设置
pre_img_path = os.path.abspath(r"D:\Search\PRT\train\yanmo\2001\yanmo1\yanmo_1\img_png")
out_path = os.path.abspath(r"D:\Search\PRT\train\yanmo\2001\yanmo1\yanmo_1\img_png_out\UMnet4")
os.makedirs(out_path, exist_ok=True)


def predict(net, device):
    net.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.363, 0.382, 0.351], std=[0.187, 0.177, 0.167])
    ])

    loader = testDataset(pre_img_path)
    test_data = DataLoader(loader, batch_size=1, shuffle=False)

    with torch.no_grad():
        for img_tensor, image_name in test_data:
            # 获取原始图像（未归一化的）
            original_img = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
            original_img = (original_img * 255).astype(np.uint8)  # 反归一化到0-255

            # 创建白色区域掩膜
            white_mask = np.all(original_img == [255, 255, 255], axis=-1)

            # 网络预测
            image = img_tensor.to(device)
            outputs = net(image)
            pred_mask = torch.argmax(outputs, dim=1).cpu().numpy()[0]  # [H, W], 值为0或1

            # 创建彩色预测图
            colored_pred = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
            colored_pred[pred_mask == 0] = PALETTE[0]  # 火灾区域 - 黄色
            colored_pred[pred_mask == 1] = PALETTE[1]  # 损失区域 - 蓝色

            # 恢复白色背景
            colored_pred[white_mask] = PALETTE[255]

            # 保存结果
            base_name = os.path.splitext(os.path.basename(image_name[0]))[0]

            # 保存彩色预测图
            Image.fromarray(colored_pred).save(os.path.join(out_path, f"{base_name}.jpg"))

            # 保存单通道掩膜图（可选）
            single_channel = np.zeros_like(pred_mask, dtype=np.uint8)
            single_channel[pred_mask == 0] = 1  # 火灾
            single_channel[pred_mask == 1] = 2  # 损失
            single_channel[white_mask] = 0  # 背景
            Image.fromarray(single_channel).save(os.path.join(out_path, f"{base_name}.png"))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    net = UMNet(3, 2)  # 输入通道3，输出通道2（火灾和损失）
    # net = FasterNet(3,2)
    state_dict = torch.load(r'D:\Search\PRT\train\weight\Unet_4.pth', map_location=device)
    net.load_state_dict(state_dict['model_state_dict'])
    net.to(device)

    predict(net, device)
