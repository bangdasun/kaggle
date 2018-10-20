### TGS Salt Identification Challenge

#### Competition Introduction

This competitions is about [Image Segmentation](https://en.wikipedia.org/wiki/Image_segmentation). The image are seismic images of subsurface, labeled with masks to indicate the existence of salt.

This is my first CV competition, due to limited time (join the competition 10 days before ending) and limited competition resource (only have laptops and no GPU / GCP credit), I have to start from kernels and discussions. I'm new to CV therefore there is a long way for me, this time I think learn to use keras / tensorflow / pytorch to implement neural nets is more important. I'll update competition summary when I have time.

#### About My Solution (370/3291)

All my works are based on kernels and discussions, essentially I just do: hyper-parameter tuning / run 5-folds / ensemble and blending. Here are my reference kernels and discussions:

- [Unet with simple ResNet blocks](https://www.kaggle.com/abhilashawasthi/unet-with-simple-resnet-blocks)
- [Introduction to U-net with simple Resnet Blocks](https://www.kaggle.com/deepaksinghrawat/introduction-to-u-net-with-simple-resnet-blocks)
- [Improving from 0.78 to 0.84+](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65226)
- [Improving from 0.84 to 0.86](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/66568)
- [+0.01 LB with snapshot ensembling and cyclic lr](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65347)

And there are more sharing after competition ending:

- [Unet+ResNetBlock+Hypercolumn+Deep supervision+Fold](https://www.kaggle.com/youhanlee/unet-resnetblock-hypercolumn-deep-supervision-fold)
- [Getting 0.87+ on Private LB using Kaggle Kernel](https://www.kaggle.com/meaninglesslives/getting-0-87-on-private-lb-using-kaggle-kernel)
- [Kent AI Lab | Ding Han | Renan 5th place solution](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69051)
- [9th place solution with code: single model](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69053)
- [11th place solution writeup](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69093)
- [32nd: strong baseline with no post-processing](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69067)
- [Vishnu: 22nd place solution along with code using Fast.ai](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69101)
- [ods.ai Power Fist solution and data explanation (11th public -> 43rd private)](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69039)
- [54th Solution: minimum post-processing](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69060)
- [126th place simple solution overview with code](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69095)
- [Noob Solution (#136)](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/69052)

More on competition kernel and discussion sections: [TGS Salt Identification Challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge)

Thanks for reading.

2018.10.20