import os
import timm
import torch
import pickle
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


def get_all_images_embeds(device, split_masks_path, split_imgs_path, output_embeds_path, encoder_model_name, crop=False):
    def get_embeds(images):
        processed_tensors = []
        for img in images:
            processed_tensors.append(transforms(img))

        output = model(torch.stack(processed_tensors).to(device))  # output is (batch_size, num_features) shaped tensor
        return output

    def crop_image(img, mask):
        #check if mask is all False
        if np.all(mask == False):
            return img, mask
        # get the bounding box of the mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        # crop the image using the bounding box and mask
        img = img[rmin:rmax+1, cmin:cmax+1]
        mask = mask[rmin:rmax+1, cmin:cmax+1]
        return img, mask

    def get_masked_images(pkl_path, img_path, level, crop=False):
        with open(pkl_path, 'rb') as f:
            masks = pickle.load(f)
        origimg = Image.open(img_path).convert('RGB').resize((224, 224))
        masked_imgs = []
        for mask in masks:
            if crop:
                img, mask2 = crop_image(np.array(origimg), mask[level])
            else:
                img = origimg
                mask2 = mask[level]
            masked = np.einsum('ijk,ij -> ijk',img, mask2) 
            masked_imgs.append(Image.fromarray(masked))
        return masked_imgs

    model = timm.create_model(
        encoder_model_name,
        pretrained=True,
        num_classes=0,  # remove classifier nn.Linear
    )
    model = model.eval()
    model.to(device)
    # get model specific transforms (normalization, maxvit_rmlp_pico_rw_256.sw_in1kresize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    for class_path in tqdm(os.listdir(split_imgs_path)):
        class_imgs_path = os.path.join(split_imgs_path, class_path)
        class_masks_path = os.path.join(split_masks_path, class_path)
        for img in tqdm(os.listdir(class_imgs_path)):
            masks_path = os.path.join(class_masks_path, img.replace('.JPEG', '.pkl'))
            img_path = os.path.join(class_imgs_path, img)
            all_levels_embeds = []
            for level in range(3):
                imgs = get_masked_images(masks_path,img_path,level, crop=crop)
                # get embeds for each batch of images
                level_embeds = []
                for i in range(0, len(imgs), 32):
                    embeds = get_embeds(imgs[i:i+32])
                    level_embeds.extend(embeds.cpu().detach().numpy())

                all_levels_embeds.append(level_embeds)
            
            img_embeds = np.stack(all_levels_embeds, axis=1)

            # save level_embeds to a file
            embeds_path = os.path.join(output_embeds_path, class_path, img.replace('.JPEG', '.pkl'))
            #mkdir if it doesnt exist
            if not os.path.exists(os.path.dirname(embeds_path)):
                os.makedirs(os.path.dirname(embeds_path))
            with open(embeds_path, 'wb') as f:
                pickle.dump(img_embeds, f)

if __name__ == '__main__':
    device='cuda:1'
    split = 'val'
    split_masks_path = f'/home/ahmedm04/projects/Agglomerator/datasets/imagenette2/{split}_masks_refined'
    split_imgs_path = f'/home/ahmedm04/projects/Agglomerator/datasets/imagenette2/{split}'
    output_embeds_path = f'/home/ahmedm04/projects/Agglomerator/datasets/imagenette2/{split}_masks_embeds_refined' 
    encoder_model_name = 'maxvit_rmlp_pico_rw_256.sw_in1k'

    parser = argparse.ArgumentParser(description='Generate embeddings for SAM Masks')    
    parser.add_argument('--device',
                        type=str,
                        default='cuda:1',
                        help='Device to run the model on')        
    
    parser.add_argument('--in_masks_path',
                        type=str,
                        help='Path to the input masks')
    
    parser.add_argument('--in_images_path',
                        type=str,
                        help='Path to the input images')
    
    parser.add_argument('--out_data_path',
                        type=str,
                        help='Path to the output embeddings data')
    
    parser.add_argument('--encoder_model_name',
                        type=str,
                        default='maxvit_rmlp_pico_rw_256.sw_in1k',
                        help='Timm encoder model name used to embed the images')
    

    parser.add_argument('--crop',
                        type=lambda x: (str(x).lower() == 'true'),
                        default=False,
                        help='Whether to crop the masked image or not')

    args = parser.parse_args()

    get_all_images_embeds(args.device, args.in_masks_path, args.in_images_path, args.out_data_path, args.encoder_model_name, args.crop)



