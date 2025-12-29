import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os.path as osp
from torchvision import transforms


# transform_train = transforms.Compose([
#     # 随机水平翻转图像和标签，概率为0.5
#     transforms.RandomHorizontalFlip(p=0.5),
#     # 随机垂直翻转图像和标签，概率为0.5
#     transforms.RandomVerticalFlip(p=0.5),
#     # 将图像和标签转换为张量，并归一化到[0,1]范围内
#     transforms.ToTensor(),
#
#     transforms.Normalize(mean=[0.369, 0.386, 0.357], std=[0.242, 0.232, 0.233])  # 标准化
# ])
# from utils.resize import keep_image_size_open

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


transform_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

transform_mask = MaskToTensor()


# class GIDDataset(Dataset):
#     def __init__(self, imgs_dir, masks_dir, train=None):
#         super(GIDDataset, self).__init__()
#         self.class_names = three_classes()
#         self.imgs_dir = imgs_dir
#         self.masks_dir = masks_dir
#         self.train = train
#
#
#         self.img_names = os.listdir(imgs_dir)
#         self.mask_names = os.listdir(masks_dir)
#
#     def __len__(self):
#         return len(self.img_names)
#
#     def __getitem__(self, i):
#         img_name = self.img_names[i]
#         img_path = osp.join(self.imgs_dir, img_name)
#         mask_path = osp.join(self.masks_dir, img_name.replace('jpg','png'))
#         image = Image.open(img_path).convert('RGB')
#         mask = Image.open(mask_path).convert('L')
#
#         mask_file_name = os.path.splitext(os.path.basename(mask_path))[0]
#         image = transform_image(image)
#         mask = transform_mask(mask)
#
#         if self.train:
#             if torch.rand(1) < 0.5:
#                 image = transforms.ColorJitter(0.4)(image)
#             # 在训练集中，图像和标签一起进行水平翻转和垂直翻转
#             if torch.rand(1) < 0.5:
#                 image = transforms.functional.hflip(image)
#                 mask = transforms.functional.hflip(mask)
#             if torch.rand(1) < 0.5:
#                 image = transforms.functional.vflip(image)
#                 mask = transforms.functional.vflip(mask)
#             return image, mask, mask_file_name
#         else:
#             return image, mask, mask_file_name
class GIDDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir, train=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.train = train
        self.image_paths = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir)]
        self.mask_paths = [os.path.join(mask_dir, fname.replace('.jpg', '.png'))
                           for fname in os.listdir(img_dir)]  # 确保图像和掩膜文件名匹配

        self.class_names = ['火灾', '非火']  # 仅两个有效类别

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # 加载图像和掩膜
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # 处理掩膜：
        mask_np = np.array(mask, dtype=np.int64)

        # if idx == 0:
        #     print("原始 mask 唯一值（预处理前）:", np.unique(mask_np))  # 应该输出 [0, 1, 2]

        new_mask = np.full_like(mask_np, -100)  # 背景设为-100（忽略）
        new_mask[mask_np == 1] = 0  # 火灾 -> 类别0
        new_mask[mask_np == 2] = 1  # 损失 -> 类别1

        # if idx == 0:
        #     print("处理后 mask 唯一值:", np.unique(new_mask))  # 应该输出 [-100, 0, 1]

        # 图像标准化
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])(img)

        # 转换为Tensor
        mask_tensor = torch.from_numpy(new_mask).long()

        # 训练模式下的数据增强
        # if self.train:
        #     if torch.rand(1) < 0.5:
        #         img = transforms.ColorJitter(brightness=0.4, contrast=0.4)(img)
        #     if torch.rand(1) < 0.5:
        #         img = transforms.functional.hflip(img)
        #         mask_tensor = transforms.functional.hflip(mask_tensor)
        #     if torch.rand(1) < 0.5:
        #         img = transforms.functional.vflip(img)
        #         mask_tensor = transforms.functional.vflip(mask_tensor)

        return img, mask_tensor


class HuaweiDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, train=None):
        super(HuaweiDataset, self).__init__()
        self.class_names = nine_classes()
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.train = train
        # self.counter = 0

        self.img_names = os.listdir(imgs_dir)
        self.mask_names = os.listdir(masks_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, i):
        img_name = self.img_names[i]
        img_path = osp.join(self.imgs_dir, img_name)
        mask_path = osp.join(self.masks_dir, img_name.replace('.tif', '.png'))
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        mask_file_name = os.path.splitext(os.path.basename(mask_path))[0]
        image = transform_image(image)
        mask = transform_mask(mask)

        if self.train:
            if torch.rand(1) < 0.5:
                image = transforms.ColorJitter()(image)
            # 在训练集中，图像和标签一起进行水平翻转和垂直翻转
            if torch.rand(1) < 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
            if torch.rand(1) < 0.5:
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)
            return image, mask, mask_file_name
        else:
            return image, mask, mask_file_name


class testDataset(Dataset):
    def __init__(self, image_path):
        self.image_path = image_path

        self.image_names = os.listdir(image_path)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        img_path = os.path.join(self.image_path, image_name)
        image = Image.open(img_path).convert('RGB')
        return transform_image(image),image_name

# class SARI_Loss(Dataset):
#     def __init__(self, imgs_dir, masks_dir, train=None):
#         super(SARI_Loss, self).__init__()
#         self.class_names = eight_classes()
#         self.imgs_dir = imgs_dir
#         self.masks_dir = masks_dir
#         self.train = train
#         # self.counter = 0
#
#         self.img_names = os.listdir(imgs_dir)
#         self.mask_names = os.listdir(masks_dir)
#
#     def __len__(self):
#         return len(self.img_names)
#
#     def __getitem__(self, i):
#         img_name = self.img_names[i]
#         img_path = osp.join(self.imgs_dir, img_name)
#         mask_path = osp.join(self.masks_dir, img_name.replace('.jpg', '.png'))
#         image = Image.open(img_path).convert('RGB')
#         mask = Image.open(mask_path).convert('L')
#
#         # label = np.array(mask).astype(np.int64)
#         # sp_idx = label[:, :, 0] + label[:, :, 1] * 256
#         # sp_idx = np.array(sp_idx)[np.newaxis, :]
#
#         mask_file_name = os.path.splitext(os.path.basename(mask_path))[0]
#
#         if self.train:
#             # 在训练集中，图像和标签一起进行水平翻转和垂直翻转
#             if torch.rand(1) < 0.5:
#                 image = transforms.functional.hflip(image)
#                 mask = transforms.functional.hflip(mask)
#             if torch.rand(1) < 0.5:
#                 image = transforms.functional.vflip(image)
#                 mask = transforms.functional.vflip(mask)
#             return transform_image(image), transform_mask(mask), mask_file_name,np.array(mask)
#         else:
#             return transform_image(image), transform_mask(mask), mask_file_name,np.array(mask)


def three_classes():
    return [
        'background',
        'loss',
        'fire'
    ]

if __name__ == '__main__':
    data1 = GIDDataset(r"/data/home2/val/image5",r"/data/home2/val/mask5")
    dataloader = DataLoader(data1,batch_size=2)
    for i, (ls, ls_msk) in enumerate(dataloader):
        print(ls.shape,ls_msk.shape)

    dataset = GIDDataset(r"/data/home2/val/image5",r"/data/home2/val/mask5")
    img, mask = dataset[0]
    # print("原始 mask 唯一值:", np.unique(mask.numpy()))
