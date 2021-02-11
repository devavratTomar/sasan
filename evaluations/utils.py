import numpy as np

colors = np.array([
    [0,     0,   0],
    [254, 232,  81], #LV-myo
    [145, 193,  62], #LA-blood
    [ 29, 162, 220], #LV-blood
    [238,  37,  36]]) #AA

def overlay_seg_img(img, seg):
    # get unique labels
    seg = seg.astype(int)
    labels = np.unique(seg)

    # remove background
    labels = labels[labels !=0]

    # img backgournd
    img_b = img*(seg == 0)

    # final_image
    final_img = np.zeros([img.shape[0], img.shape[1], 3])

    final_img += img_b[:, :, np.newaxis]

    for l in labels:
        mask = seg == l
        img_f = img*mask

        # convert to rgb
        img_f = np.tile(img_f, (3, 1, 1)).transpose(1, 2, 0)

        # colored segmentation
        img_seg = colors[l*mask]

        # alpha overlay
        final_img += 0.5*img_f + 0.5*img_seg
    
    return final_img