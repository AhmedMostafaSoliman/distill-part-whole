import os
import cv2
import tqdm
import torch
import pickle 
import argparse
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide


def prepare_image(image, transform, device):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device.device) 
    return image.permute(2, 0, 1).contiguous()

def get_patched_output(dataset_dir, out_data_path, patch_size, img_size, processing_batch_size=5):
    for image_class in os.listdir(dataset_dir):
        if not os.path.exists(os.path.join(out_data_path, image_class)):
                os.makedirs(os.path.join(out_data_path, image_class))  

        class_dir = os.path.join(dataset_dir, image_class)
        class_image_paths = os.listdir(class_dir)
        class_chunk_batched_input = []
        imgpath_chunk = []

        for image in tqdm.tqdm(class_image_paths):
            image_path = os.path.join(class_dir, image)
            img_extension = image_path.split('.')[-1]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (img_size,img_size))
            image_points = [[[y,x]] for x in range(patch_size//2,img_size,patch_size) for y in range(patch_size //2,img_size,patch_size)]
            point_labels = [[i] for i in range(len(image_points))]
            image_points = torch.tensor(image_points, device=sam.device)
            point_labels = torch.tensor(point_labels, device=sam.device)
            image_dict = {
                'image': prepare_image(image, resize_transform, sam),
                'point_coords': resize_transform.apply_coords_torch(image_points, image.shape[:2]),
                'point_labels': point_labels,
                'original_size': image.shape[:2]
            }
            class_chunk_batched_input.append(image_dict)
            imgpath_chunk.append(image_path)
            if len(class_chunk_batched_input)% processing_batch_size == 0 and len(class_chunk_batched_input):
                class_chunk_batched_output = sam(class_chunk_batched_input, multimask_output=True)
                for img_name, item in zip(imgpath_chunk,class_chunk_batched_output):
                    out_f_path = os.path.join(out_data_path, image_class, img_name.split('/')[-1].replace('.'+img_extension,'.pkl'))
                    with open(out_f_path, 'wb') as f:
                        pickle.dump([mask.cpu().numpy() for mask in item['masks']], f, protocol=pickle.HIGHEST_PROTOCOL)

                # clear the processed batch data
                class_chunk_batched_input= []
                imgpath_chunk = []
                del class_chunk_batched_output
                torch.cuda.empty_cache()

        # process the remaining batch
        if(len(class_chunk_batched_input)):
            class_chunk_batched_output = sam(class_chunk_batched_input, multimask_output=True)
            for img_name, item in zip(imgpath_chunk,class_chunk_batched_output):
                out_f_path = os.path.join(out_data_path, image_class, img_name.split('/')[-1].replace('.'+img_extension,'.pkl'))
                with open(out_f_path, 'wb') as f:
                    pickle.dump([mask.cpu().numpy() for mask in item['masks']], f, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    # sam_checkpoint = "sam_vit_h_4b8939.pth"
    # model_type = "vit_h"
    # device = "cuda:1"
    # patch_size = 16
    # img_size = 224
    # split = 'val'
    # in_data_path = '/home/ahmedm04/projects/distill_part_whole/datasets/imagenette2/val'
    # out_data_path = '/home/ahmedm04/projects/distill_part_whole/datasets/imagenette2/val_masks'

    parser = argparse.ArgumentParser(description='Generate SAM Masks for the dataset')
    parser.add_argument('--sam_checkpoint',
                        type=str,
                        default='sam_vit_h_4b8939.pth', 
                        help='Path to the SAM checkpoint')
    
    parser.add_argument('--model_type',
                        type=str,
                        default='vit_h',
                        help='Type of the model')
    
    parser.add_argument('--device',
                        type=str,
                        default='cuda:1',
                        help='Device to run the model on')

    parser.add_argument('--patch_size',
                        type=int,
                        default=16,
                        help='Patch size to use for the model')
    
    parser.add_argument('--img_size',
                        type=int,
                        default=224,
                        help='Image size to use for the model')
    
    parser.add_argument('--split',
                        type=str,
                        default='val',
                        help='Split to use for the model')
    
    parser.add_argument('--in_data_path',
                        type=str,
                        help='Path to the input Images')
    
    parser.add_argument('--out_data_path',
                        type=str,
                        help='Path to the output data')
    
    args = parser.parse_args()

    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device)    

    predictor = SamPredictor(sam)
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
    
    get_patched_output(args.in_data_path, args.out_data_path, args.patch_size, args.img_size, processing_batch_size=5)
