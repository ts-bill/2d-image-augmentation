# 2d-image-augmentation

#### Resource
- imgaug library
    - https://github.com/aleju/imgaug
- รวมงานด้าน image augmentation
    - https://github.com/CrazyVertigo/awesome-data-augmentation
#### Requirement
- Python > 3.8
- Opencv
- imgaug
- tqdm
- imageio
- 
#### How to use
1. Change image and output path in ``img_aug_with_label.py``
    - ``im_path``: input image path *example = "data/images/"*
    - ``txt_path``: annotation for input image path *example = "data/lables/"*
    - ``output_path``: output image that passed image augmentation *example = "output/images/"*
    - ``lables_output_path``: annotation for output image that passed image augmentation *example = "output/lables/"*
    - ``show_output_path``: example of output images will save here *example = "output/show/"*
2. run command ``python img_aug_with_lable.py``