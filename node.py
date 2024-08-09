from .semantic_sam import build_semantic_sam, SemanticSAMPredictor
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import ToPILImage
import scipy
import sys, os
import logging
import cv2
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

sam_model_list = {"L": {"model_url": "custom_nodes/ComfyUI_Semantic_SAM/ckpt/swinl_only_sam_many2many.pth"}, 
                  "T": {"model_url": "custom_nodes/ComfyUI_Semantic_SAM/ckpt/swint_only_sam_many2many.pth"}}

def list_sam_model():
    return list(sam_model_list.keys())

def sort_masks(masks, ious, thresh):
    ious = ious[0, 0]
    ids = torch.argsort(ious, descending=True)
    sort_masks = []
    for i, (mask, iou) in enumerate(zip(masks[ids], ious[ids])):
        iou = round(float(iou), 2)
        mask = mask.cpu().detach().numpy()
        if iou < thresh:
            continue
        mask[mask<=0]=0
        mask[mask>0]=255
        mask = mask.astype(np.uint8)
        mask = mask/255
        sort_masks.append(mask)
    return sort_masks


class PointPrompt():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x_coord": ("INT",{}),
                "y_coord": ("INT",{})
            },
            "optional": {}
        }
        
    CATEGORY = "SemanticSAM"

    RETURN_TYPES = ("POINTS",)

    FUNCTION = "main"
    
    def main(self,  x_coord, y_coord):
        return ([[x_coord,x_coord]],)
    
class SemanticSAMLoader():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list_sam_model(),)
                
            },
            "optional": {}
        }
        
    CATEGORY = "SemanticSAM"

    RETURN_TYPES = ("SemanticSAM_Model",)

    FUNCTION = "main"
    
    def main(self,  model_name):
        ckpt_path = sam_model_list[model_name]["model_url"]
        mask_generator = SemanticSAMPredictor(build_semantic_sam(model_type=model_name, ckpt=ckpt_path))
        return (mask_generator,)
    
class SemanticSAMSegment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SemanticSAM_Model", {}),
                "image": ("IMAGE", {}),
                "points": ("POINTS", {}),
                "expand": ("INT", {"default":0, "min":-10, "max":10, "step":1}),
                "threshold": ("FLOAT", {"default":0.5, "min":0, "max":1, "step": 0.01}),
                "num_masks": ("INT", {"default":6, "min":1, "max":6, "step":1})
            },
            "optional": {}
        }
    
    CATEGORY = "SemanticSAM"

    RETURN_TYPES = ("IMAGE","MASK")

    FUNCTION = "main"

    def main(self, model, image, points: list, expand: int, threshold: float=0.3, num_masks: int=0):
        # image [BxHxWxC]
        original_image, input_image = self.prepare_image(image)
        #print(image.shape, original_image.shape, input_image.shape)
        
        h = image.shape[1]
        w = image.shape[2]
        # 坐标归一化
        points = np.array(points).astype(np.float32)
        #print(points, [:,0],points[:,1])
        points[:, 0] = points[:, 0]/w
        points[:, 1] = points[:, 1]/h
        #print(points)
        # 分割
        masks, ious = model.predict(original_image, input_image, point=points)
        
        masks = masks[0:num_masks]
        # sort_mask, np.array, pixel range: [0,1], 二值图
        # sort_masks, list
        sorted_masks = sort_masks(masks, ious, threshold)
        if expand!=0:
            sorted_masks = expand_mask(sorted_masks, expand=expand)
        rgba_imgs, masks_tensor = masks2rgba(image[0], sorted_masks)
        #rint(sorted_masks)
        #masks_tensor = torch.from_numpy(sorted_masks)
        return (torch.cat(rgba_imgs, dim=0), masks_tensor)
    
    def prepare_image(self, image: torch.tensor):
        """_summary_
        Args:
            image (torch.tensor): [BxHxWxC]

        Returns:
            image_ori (np.array): [HxWxC]
            images (torch.tensor): [CxHxW]
        """
        #image = Image.fromarray(image[0].numpy().astype('uint8'))
        t = []
        image = image[0].permute(2,0,1) #[B,H,W,C] -> [C,H,W]
        #print(image.shape)
        t.append(ToPILImage())
        t.append(transforms.Resize(640, interpolation=Image.BICUBIC))
        transform1 = transforms.Compose(t)
        image_ori = transform1(image)
        
        image_ori = np.asarray(image_ori) # [HxWxC]
        #print(image_ori.shape)
        images = torch.from_numpy(image_ori.copy()).permute(2,0,1).cuda() #permute(2,0,1) CxHxW
        #print(images.size())
        
        return image_ori, images
    
def expand_mask(mask, expand=0, tapered_corners=True):
    """對masks進行膨脹或者腐蝕

    Args:
        mask (list[np.array]): mask list
        expand (int, optional): （扩展大小. Defaults to 0.
        tapered_corners (bool, optional): （是否使用锥形角）. Defaults to True.

    Returns:
        (list[np.array]): processed mask list
    """
    mask = np.array(mask.astype(np.uint8))
    c = 0 if tapered_corners else 1
    kernel = np.array([[c, 1, c],
                        [1, 1, 1],
                        [c, 1, c]])
    mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
    out = []
    for m in mask:
        output = m
        for _ in range(abs(expand)):
            if expand < 0:
                output = scipy.ndimage.grey_erosion(output, footprint=kernel)
            else:
                output = scipy.ndimage.grey_dilation(output, footprint=kernel)
        output = np.array(output)
        out.append(output)
    return out

def masks2rgba(image, masks):
    rgba_imgs = []
    masks_tensor = []
    for i, mask in enumerate(masks):
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        rgba_tensor = mask2rgba(image, mask)
        rgba_imgs.append(rgba_tensor)
        #torch.unsqueeze(mask, dim=0)
        masks_tensor.append(torch.from_numpy(mask).unsqueeze(dim=0))
    masks_tensor = torch.cat(masks_tensor, dim=0)
    return rgba_imgs, masks_tensor

def mask2rgba(image, mask):
    #print("mask shape:",mask.shape)
    rgba_image = np.zeros((mask.shape[0], mask.shape[1], 4))
    # 将 RGB 图像复制到 RGBA 图像的前三个通道
    rgba_image[:, :, :3] = image.numpy().astype(np.float32)
    # 将 mask 应用到 alpha 通道，其中 mask 为 255（白色）的部分设为完全不透明（255），其他部分设为完全透明（0）
    rgba_image[:, :, 3] = np.where(mask == 1, 1, 0)
    #cv2.imwrite(f'mask_{i}.png', rgba_image*255)
    rgba_tensor = torch.from_numpy(rgba_image)
    rgba_tensor = torch.unsqueeze(rgba_tensor, dim=0)
    return rgba_tensor