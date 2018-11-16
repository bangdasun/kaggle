### Airbus Ship Detection Challenge

#### Competition Introduction

This competitions is about Object Detection and Image Segmentation. The image are satellite images which might contains ships, we need to predict the mask of the ships instead of drawing bounding box on ships.

This is my second CV competition, again due to limited competition resource (only have laptops and no GPU / GCP credit), I have to start from kernels and discussions. I'll update competition summary when I have time.

#### About My Solution (121/884)

All my works are based on kernels and discussions, essentially I just do: hyper-parameter tuning. This time I get started with fast.ai, which is a high-level package like keras. Also learn more basic stuff about computer vision than TGS Competition. Here are my reference kernels and discussions:

- [Unet34 submission TTA (0.699 new public LB)](https://www.kaggle.com/iafoss/unet34-submission-tta-0-699-new-public-lb)
- [Fine-tuning ResNet34 on ship detection (new data)](https://www.kaggle.com/iafoss/fine-tuning-resnet34-on-ship-detection-new-data)
- [Airbus EDA](https://www.kaggle.com/ezietsman/airbus-eda)
- [Airbus Ship Detection: Data Visualization](https://www.kaggle.com/meaninglesslives/airbus-ship-detection-data-visualization)
- [2 - Understanding and plotting rle bounding boxes](https://www.kaggle.com/julian3833/2-understanding-and-plotting-rle-bounding-boxes)
- [Airbus Mask-RCNN and COCO transfer learning](https://www.kaggle.com/hmendonca/airbus-mask-rcnn-and-coco-transfer-learning)
- [From masks to bounding boxes](https://www.kaggle.com/voglinio/from-masks-to-bounding-boxes)

And there are more sharing after competition ending:

- [Few lessons learned (4th place)](https://www.kaggle.com/c/airbus-ship-detection/discussion/71667)
- [Do not trust the LB, trust your CV. (5th/8th in public/private LB)](https://www.kaggle.com/c/airbus-ship-detection/discussion/71601)
- [9th place solution](https://www.kaggle.com/c/airbus-ship-detection/discussion/71595#latest-422251)
- [10th MaskRCNN without ensemble and TTA solution.](https://www.kaggle.com/c/airbus-ship-detection/discussion/71607#latest-421764)
- [11th place solution](https://www.kaggle.com/c/airbus-ship-detection/discussion/71659#latest-422049)
- [14th place solution: data and metric comprehension](https://www.kaggle.com/c/airbus-ship-detection/discussion/71664#latest-422079)

More on competition kernel and discussion sections: [Airbus Ship Detection Challenge](https://www.kaggle.com/c/airbus-ship-detection).

Thanks for reading.

2018.11.15