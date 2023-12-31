# Distilling Part-whole Hierarchical Knowledge from a Huge Pretrained Class Agnostic Segmentation Framework
Official code for "Distilling Part-whole Hierarchical Knowledge from a Huge Pretrained Class Agnostic Segmentation Framework" ([Ahmed Radwan](https://scholar.google.com/citations?user=LCz8YhMAAAAJ&hl=en), [Mohamed S. Shehata](https://scholar.google.com/citations?hl=en&user=i9PpMVkAAAAJ)) (Accepted for ICCV'23's Workshop Visual Inductive Priors for Data-Efficient Deep Learning) \[[pdf](https://openaccess.thecvf.com/content/ICCV2023W/VIPriors/papers/Radwan_Distilling_Part-Whole_Hierarchical_Knowledge_from_a_Huge_Pretrained_Class_Agnostic_ICCVW_2023_paper.pdf)\]

## Requirements Setup

From the main directory run:

``pipenv install``

to install the required dependencies for the Agglomerator.

## SAM Setup

``pip install git+https://github.com/facebookresearch/segment-anything.git``

## Dataset

We performed our experiments on Imagenette. Download the full size version from [here] (https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz)
then place it in datasets/imagenette2

## Generating SAM Masks

Firstly, we will generate the multi-output SAM masks for our dataset with the desired configurations.
For examples to generate the masks for the validation set and for the particular configuration defined in ``config/config_IMAGENETTE.cfg`` we would like to divide an image with size 224 x 224 into 14 x 14 (i.e patches each with size 16 x 16), so we execute:

``python src/hierarchical_masks/patched_masks.py --device=cuda:1 --split=val --in_data_path=path_to_val_input_images --out_data_path=path_to_out_val_masks --sam_checkpoint=path_to_vit_h_sam_checkpoint --model_type=vit_h --img_size 224 --patch_size=16``

this will generate a pickled file for each image containing 14 x 14 x 3 masks

## Generating Masks embeddings

After generating the masks, we need to generate the embeddings for them. We use an encoder that matches the desired patch dimension. For example, to generate the embeddings without cropping and also to match the patch_dim in the config file in ``config/config_IMAGENETTE.cfg`` we might use the pretrained model ``maxvit_rmlp_pico_rw_256.sw_in1k`` to generate the embeddings for the validation set, we execute:

``python src/hierarchical_masks/masks_to_embeds.py --in_masks_path=path_to_the_sam_generated_masks --in_images_path=path_to_images_used_to_gen_sam_masks --out_data_path=project_root_path/datasets/imagenette2/val_masks_embeds --encoder_model_name='maxvit_rmlp_pico_rw_256.sw_in1k'``


## Training / Validation

Update the configuration file with the desired parameters then, for example, to run pre-training on Imagenette on GPU 0, execute:

``CUDA_VISIBLE_DEVICES=0 python src/main.py --flagfile config/config_IMAGENETTE.cfg``

After running the pre-training you can run the training phase with:

``CUDA_VISIBLE_DEVICES=0 python src/main.py --flagfile config/config_IMAGENETTE.cfg --resume_training --supervise --load_checkpoint_dir <path_to_checkpoint.ckpt>``

To run testing or to freeze the network weights, set the 'mode' flag (e.g. ``--mode test`` or ``--mode freeze``). 
Refer to [this page](src/flags_Agglomerator.py) for additional info about each flag.

![Training](img/SAM_pretrain.png)

## Plotting islands of agreement

To enable live visualization of the islands of the agreement during training/val/test, set the flag ``--plot_islands``.

![Islands](img/islands_viz.jpg)

## Citing

    @InProceedings{Radwan_2023_ICCV,
        author    = {Radwan, Ahmed and Shehata, Mohamed S.},
        title     = {Distilling Part-Whole Hierarchical Knowledge from a Huge Pretrained Class Agnostic Segmentation Framework},
        booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
        month     = {October},
        year      = {2023},
        pages     = {238-246}
    }

## Credits

- Theoretical idea by [Geoffrey Hinton](https://arxiv.org/pdf/2102.12627.pdf)
- Agglomerator model by [mmlab-cv](https://github.com/mmlab-cv/Agglomerator)
- Segmentation framework by [FAIR](https://ai.meta.com/research/publications/segment-anything)
- Pretrained models for generating the masks embedding by [Ross Wightman](https://github.com/huggingface/pytorch-image-models)
