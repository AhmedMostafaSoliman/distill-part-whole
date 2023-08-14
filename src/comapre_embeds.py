import pickle
import einops
import numpy as np



sample_embeds1 = pickle.load(open('/home/ahmedm04/projects/distill_part_whole/datasets/imagenette2/val_masks_embeds/n03028079/ILSVRC2012_val_00003351.pkl', 'rb'))
dim = int(np.sqrt(sample_embeds1.shape[0]))
sample_embeds1 = einops.rearrange(sample_embeds1, '(h w) l p -> (w h) l p', h=dim, w=dim) 

sample_embeds2 = pickle.load(open('/home/ahmedm04/projects/distill_part_whole/datasets/imagenette2/val_masks_embeds_2/n03028079/ILSVRC2012_val_00003351.pkl', 'rb'))

# compare the two embeds
for i in range(len(sample_embeds1)):
    print("Embedding {}:".format(i))
    print("Embed 1: {}".format(sample_embeds1[i].shape))
    print("Embed 2: {}".format(sample_embeds2[i].shape))
    print("Embed 1 - Embed 2: {}".format((sample_embeds1[i] - sample_embeds2[i]).sum()))

