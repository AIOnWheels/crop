import cv2
import os
import glob
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

BATCH_SIZE = 64
IMG_DIR = './images/train/'


class CropDataset(Dataset):
    def __init__(self, file_path, transform=None):
        xform = transforms.Compose([transforms.ToTensor()]) if transform is None else transform
        self.img_dataset = datasets.ImageFolder(file_path, transform=xform)

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, idx):
        image, label = self.img_dataset[idx]
        return image, label


class ProfileImage:
    def __init__(self, profile_image_path, grey_scale=False, normalize=False, batch_size=64):
        # return a tensor of the images 
        self.profile_image_path = profile_image_path
        transform = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize((0.4743617, 0.49847862, 0.4265874 ),(0.21134755, 0.19044809, 0.22679578))])
        self.img_dataset = datasets.ImageFolder(self.profile_image_path, transform=transform)
        self.img_dataloader = DataLoader(self.img_dataset, batch_size=1, shuffle=False)

    def get_image_distribution(self, profile_image_path):
        imgs = []
        print(glob.glob('./images/train/Apple__Apple_scrab/*.JPG'))
        for fn in glob.glob('./images/train/Apple__Apple_scrab/*.JPG'):
            im = cv2.imread(fn, cv2.IMREAD_COLOR)
            imgs.append(im)
        
        # for root, dirs, files in os.walk(profile_image_path):
        #     all_files = [os.path.join(root,file) for file in files]
            # for file in files:
            #     if file == '.DS_Store':
            #         pass
            #     else:
            # img = cv2.imread(all_files)
            # images.append(img.shape)

        # dirs = os.listdir(profile_image_path)
        # record = []
        # for dir in dirs:
        #     if(dir=='.DS_Store'):
        #         pass
        #     else:
        #         if(dir=='.DS_Store'):
        #             pass
        #         else:
        #             files = os.listdir(profile_image_path + dir)
        #             for file in files:
        #                 img = cv2.imread(profile_image_path + dir + '/' + file)
        #                 record.append(img.shape)
        return imgs

if __name__ == '__main__':
    profile_image = ProfileImage(IMG_DIR)
    print(len(profile_image.get_image_distribution(IMG_DIR)))

    # for i, (img, label) in enumerate(profile_image.img_dataloader):
    #     print(i, img.shape, label.shape)
    #     break;

    crop_data = CropDataset(IMG_DIR)
    image, label = crop_data.__getitem__(11)
    print(image.shape, label)