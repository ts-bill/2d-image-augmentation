import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import imageio
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import os
import cv2
from tqdm import tqdm
ia.seed(77)
im_path = "data/images/"
im_list = []
txt_path = "data/lables/"
output_path = 'output/images/'
lables_output_path = 'output/lables/'
show_output_path = 'output/show/'
GREEN = [0, 255, 0]
ORANGE = [255, 140, 0]
RED = [255, 0, 0]
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
for x in os.listdir(im_path):
    if x.endswith(".png") or x.endswith(".jpg"):
        # Prints only text file present in My Folder
        print(x)
        im_list.append(x)
aug_num = 5
def draw_bbs(image, bbs, border):
    image_border = pad(image, border)
    for bb in bbs.bounding_boxes:
        if bb.is_fully_within_image(image.shape):
            color = GREEN
        elif bb.is_partly_within_image(image.shape):
            color = ORANGE
        else:
            color = RED
        image_border = bb.shift(left=border, top=border)\
                         .draw_on_image(image_border, size=2, color=color)
def pad(image, by):
    image_border1 = ia.pad(image, top=1, right=1, bottom=1, left=1,
                           mode="constant", cval=255)
    image_border2 = ia.pad(image_border1, top=by-1, right=by-1,
                           bottom=by-1, left=by-1,
                           mode="constant", cval=0)
    return image_border2
for picname in tqdm(im_list):
    bbox = []
    cls = []
    bbs = []
    st = ''
    filename = picname.split(".")
    check_txt = os.path.exists(txt_path + filename[0] + '.txt')
    pipe_image = imageio.imread(im_path + picname)

    
    #image = np.array(pipe_image , dtype=np.uint8)
    # Example batch of images.
    # The array has shape (32, 64, 64, 3) and dtype uint8.
    images = np.array(
        [pipe_image for _ in range(aug_num)],
        dtype=np.uint8
    )
    if check_txt:
        with open(txt_path + filename[0] + '.txt') as f:
            lines = f.readlines()
        for line in lines:
            num = line.split(' ')
            cls.append(num[0])
            #print(num[0],num[1],num[2],num[3],num[4])
            bbox_w, bbox_h = float(float(num[3]) * pipe_image.shape[1]), float(float(num[4]) * pipe_image.shape[0])
            bbox_x, bbox_y = float((float(num[1])*pipe_image.shape[1]) - float(bbox_w/2)), float((float(num[2]) * pipe_image.shape[0]) - float(bbox_h/2))
            bbox.append(BoundingBox(x1=bbox_x,y1=bbox_y,x2=bbox_w + bbox_x,y2=bbox_h + bbox_y))
    
    #print(bbox, "---------------------------")
    #for icp in range(aug_num):
    #    cpbbox.append(bbox)
    #print(cpbbox)
    #bbs = BoundingBoxesOnImage([bbox for _ in range(aug_num)], shape= pipe_image.shape)
    #print(bbs)
    #bbs2 = BoundingBoxesOnImage([b for b in bbox], shape=pipe_image.shape) 
    #print(bbs2)
    #print("---------------------------")
        for _ in range(aug_num):
            bbs.append(BoundingBoxesOnImage([b for b in bbox], shape=pipe_image.shape))

    #seq = iaa.Sequential([
    #    iaa.Fliplr(0.5), # horizontal flips
    #    #iaa.Crop(percent=(0, 0.1)), # random crops
    #    # Small gaussian blur with random sigma between 0 and 0.5.
    #    # But we only blur about 50% of all images.
    #    iaa.Sometimes(
    #        0.5,
    #        iaa.GaussianBlur(sigma=(0, 0.8))
    #    ),
    #    # Strengthen or weaken the contrast in each image.
    #    iaa.LinearContrast((0.75, 1.5)),
    #    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    #    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    #    iaa.Affine(
    #        #scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
    #    ),
    #    iaa.CropToFixedSize(width=1080, height=1920)
    #    ], random_order=True) # apply augmenters in random order
    seq = iaa.Sequential(
    [
        #
        # Apply the following augmenters to most images.
        #
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        iaa.Flipud(0.2), # vertically flip 20% of all images

        # crop some of the images by 0-10% of their height/width
        sometimes(iaa.Crop(percent=(0, 0.1))),

        # Apply affine transformations to some of the images
        # - scale to 80-120% of image height/width (each axis independently)
        # - translate by -20 to +20 relative to height/width (per axis)
        # - rotate by -45 to +45 degrees
        # - shear by -16 to +16 degrees
        # - order: use nearest neighbour or bilinear interpolation (fast)
        # - mode: use any available mode to fill newly created pixels
        #         see API or scikit-image for which modes are available
        # - cval: if the mode is constant, then use a random brightness
        #         for the newly created pixels (e.g. sometimes black,
        #         sometimes white)
        sometimes(iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            #rotate=(-45, 45),
            #shear=(-16, 16),
            #order=[0, 1],
            #cval=(0, 255),
            #mode=ia.ALL
        )),

        #
        # Execute 0 to 5 of the following (less important) augmenters per
        # image. Don't execute all of them, as that would often be way too
        # strong.
        #
        iaa.SomeOf((0, 5),
            [
                # Convert some images into their superpixel representation,
                # sample between 20 and 200 superpixels per image, but do
                # not replace all superpixels with their average, only
                # some of them (p_replace).
                sometimes(
                    iaa.Superpixels(
                        p_replace=(0, 1.0),
                        n_segments=(20, 200)
                    )
                ),

                # Blur each image with varying strength using
                # gaussian blur (sigma between 0 and 3.0),
                # average/uniform blur (kernel size between 2x2 and 7x7)
                # median blur (kernel size between 3x3 and 11x11).
                iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 11)),
                ]),

                # Sharpen each image, overlay the result with the original
                # image using an alpha between 0 (no sharpening) and 1
                # (full sharpening effect).
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),

                # Same as sharpen, but for an embossing effect.
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                # Search in some images either for all edges or for
                # directed edges. These edges are then marked in a black
                # and white image and overlayed with the original image
                # using an alpha of 0 to 0.7.
                sometimes(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0, 0.7)),
                    iaa.DirectedEdgeDetect(
                        alpha=(0, 0.7), direction=(0.0, 1.0)
                    ),
                ])),

                # Add gaussian noise to some images.
                # In 50% of these cases, the noise is randomly sampled per
                # channel and pixel.
                # In the other 50% of all cases it is sampled once per
                # pixel (i.e. brightness change).
                iaa.AdditiveGaussianNoise(
                    loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                ),

                # Either drop randomly 1 to 10% of all pixels (i.e. set
                # them to black) or drop them on an image with 2-5% percent
                # of the original size, leading to large dropped
                # rectangles.
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout(
                        (0.03, 0.15), size_percent=(0.02, 0.05),
                        per_channel=0.2
                    ),
                ]),

                # Invert each image's channel with 5% probability.
                # This sets each pixel value v to 255-v.
                iaa.Invert(0.05, per_channel=True), # invert color channels

                # Add a value of -10 to 10 to each pixel.
                iaa.Add((-10, 10), per_channel=0.5),

                # Change brightness of images (50-150% of original value).
                iaa.Multiply((0.5, 1.5), per_channel=0.5),

                # Improve or worsen the contrast of images.
                iaa.LinearContrast((0.5, 2.0), per_channel=0.5),

                # Convert each image to grayscale and then overlay the
                # result with the original with random alpha. I.e. remove
                # colors with varying strengths.
                iaa.Grayscale(alpha=(0.0, 1.0)),

                # In some images move pixels locally around (with random
                # strengths).
                sometimes(
                    iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)
                ),

                # In some images distort local areas with varying strength.
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05)))
            ],
            # do all of the above augmentations in random order
            random_order=True
        )
    ],
    # do all of the above augmentations in random order
    random_order=True
)


    if check_txt:
        images_aug , bbs_aug = seq(images=images, bounding_boxes = bbs)
        for idx, img_aug in enumerate(images_aug) :
            st = ''
            for i in range(len(bbs[idx].bounding_boxes)):
                before = bbs[idx].bounding_boxes[i]
                after = bbs_aug[idx].bounding_boxes[i]
        #image = np.array(pipe_image , dtype=np[i]
                print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
                    i,
                    before.x1, before.y1, before.x2, before.y2,
                    after.x1, after.y1, after.x2, after.y2)
                )
                c_w = ((after.x2 - after.x1) *  0.592592593) / 640.0
                c_h = ((after.y2 - after.y1) * 0.333333333) / 640.0
                c_x = ((after.x1 + ((after.x2 - after.x1)/2.0)) * 0.592592593) / 640.0
                c_y = ((after.y1 + ((after.y2 - after.y1)/2.0)) * 0.333333333) / 640.0

                st += str(cls[i]) + ' ' + str(c_x) + ' ' + str(c_y) + ' ' + str(c_w) + ' ' + str(c_h) + "\n"
            with open(lables_output_path +  filename[0] + "-aug" + str(idx) + '.txt', 'w') as f:
                f.writelines(st)    
            
            image_before = bbs[idx].draw_on_image(images[idx], size=2)
            image_after = bbs_aug[idx].draw_on_image(img_aug, size=5, color=[0, 0, 255])
            #image_after = draw_bbs(img_aug, bbs_aug[idx].remove_out_of_image().clip_out_of_image(), 100)
            img_aug = cv2.resize(img_aug, (640, 640))
            image_after = cv2.resize(image_after, (640, 640))
            imageio.imwrite(output_path + filename[0] + "-aug" + str(idx) +'.png', img_aug)  #write all changed images
            imageio.imwrite(show_output_path + filename[0] + "-aug" + str(idx)+'.png', image_after)  #write all changed images 
        print("-------------------------------------------------------------------------")
        
    else:
        images_aug = seq(images=images)
        for idx, img_aug in enumerate(images_aug) :
            img_aug = cv2.resize(img_aug, (640, 640))
            imageio.imwrite(output_path + filename[0] + "-aug" + str(idx) +'.png', img_aug)
        print("-------------------------------NOTXT------------------------------------------")
    #break
#for i in range(1):
#imageio.imwrite('output/' + str(i)+'new.jpg', image_before[i])  #write all changed images
#imageio.imwrite('output/' + 'after' +str(i)+'new.jpg', image_after[i])  #write all changed images
#imageio.imwrite('output/' + 'new.jpg', image_before)  #write all changed images
#imageio.imwrite('output/' + 'after' +'new.jpg', image_after)  #write all changed images