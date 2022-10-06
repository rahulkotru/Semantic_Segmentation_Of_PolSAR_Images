import random
import numpy as np

def generate_random_patches(img, mask, sz):
    """
    :param img: ndarray with shape (x_sz, y_sz, num_channels)
    :param mask: binary ndarray with shape (x_sz, y_sz, num_classes)
    :param sz: size of random patch
    :return: patch with shape (sz, sz, num_channels)
    """
    assert len(img.shape) == 3 and img.shape[0] >= sz or img.shape[1] >= sz and img.shape[0:2] == mask.shape[0:2]
    xc = 0
    yc = 0
    patch_img = img[xc:(xc + sz), yc:(yc + sz)]
    patch_mask = mask[xc:(xc + sz), yc:(yc + sz)]

    # Apply some random transformations
    random_transformation = np.random.randint(1,8)
    if random_transformation == 1:  # reverse first dimension
        patch_img = patch_img[::-1,:,:]
        patch_mask = patch_mask[::-1,:,:]
    elif random_transformation == 2:    # reverse second dimension
        patch_img = patch_img[:,::-1,:]
        patch_mask = patch_mask[:,::-1,:]
    elif random_transformation == 3:    # transpose(interchange) first and second dimensions
        patch_img = patch_img.transpose([1,0,2])
        patch_mask = patch_mask.transpose([1,0,2])
    elif random_transformation == 4:
        patch_img = np.rot90(patch_img, 1)
        patch_mask = np.rot90(patch_mask, 1)
    elif random_transformation == 5:
        patch_img = np.rot90(patch_img, 2)
        patch_mask = np.rot90(patch_mask, 2)
    elif random_transformation == 6:
        patch_img = np.rot90(patch_img, 3)
        patch_mask = np.rot90(patch_mask, 3)
    else:
        pass
    return patch_img,patch_mask
    #return img, mask


def get_patches(x_dict, y_dict, n_patches, sz):
    x = list()
    y = list()
    total_patches = 0
    while total_patches <n_patches:# replace 0 with  n_patches
        img_id = random.sample(x_dict.keys(), 1)[0]#[0] makes the list hashable
        print(img_id)
        img = x_dict[img_id]
        mask = y_dict[img_id]
        img_patch, mask_patch = generate_random_patches( img, mask,sz)# write get_rand_patch(img,mask,sz)
        x.append(img_patch)#img_patch
        y.append(mask_patch)#mask_patch
        total_patches += 1
    print('Generated {} patches'.format(total_patches))
    return np.array(x), np.array(y)
