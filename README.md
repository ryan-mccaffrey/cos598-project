# Image Caption Validation
This repository contains the code for the COS 598B final project designed by Ryan McCaffrey and Yannis Karakozis. The architecture of the model is heavily inspired by the following paper:

* R. Hu, M. Rohrbach, T. Darrell, *Segmentation from Natural Language Expressions*. in ECCV, 2016. ([PDF](http://arxiv.org/pdf/1603.06180))
```
@article{hu2016segmentation,
  title={Segmentation from Natural Language Expressions},
  author={Hu, Ronghang and Rohrbach, Marcus and Darrell, Trevor},
  journal={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2016}
}
```

A graphic of the architecture is below:
![Model Architecture](https://github.com/ryan-mccaffrey/cos598-project/images/model-architecture.png)

## Installation
1. Install Google TensorFlow (v1.0.0 or higher) following the instructions [here](https://www.tensorflow.org/install/).
2. Download this repository or clone with Git, and then `cd` into the root directory of the repository.

## Demo
1. Download pre-trained models by contacting the authors at `{rm24,ick}@princeton.edu`.
2. Run the language-based segmentation model demo in `./demo/text_objseg_cls_glove_demo.ipynb` with [Jupyter Notebook (IPython Notebook)](http://ipython.org/notebook.html). If the demo with the learned word embedding is desired, run the demo in `./demo/text_objseg_cls_demo.ipynb`.



## Training and evaluation on COCO Dataset

### Download VGG network, GloVe embeddings, COCO files
1. Download VGG-16 network parameters trained on ImageNET 1000 classes:  
`models/convert_caffemodel/params/download_vgg_params.sh`.
2. Download the pre-trained GloVe word embeddings from the [chakin](https://github.com/chakki-works/chakin) Github repository. Follow the repository instructions to download the `glove.6B.50d.txt` and `glove.6B.300d.txt` files, and then place both in the `exp-referit/data` repository.
3. Download the testing and training COCO annotations from the [download site](http://cocodataset.org/#download). Choose the `2017 Train/Val annotations` zip, unpack it, and place the files in `coco/annotations`.

### Training
4. You may need to add the repository root directory to Python's module path: `export PYTHONPATH=.:$PYTHONPATH`.
5. Build training batches:  
`python coco/build_training_batches_cls_coco.py`. Check the file to see different arguments that can be given to change how files are generated, which will impact how the model trains.
6. Select the GPU you want to use during training:  
`export GPU_ID=<gpu id>`. Use 0 for `<gpu id>` if you only have one GPU on your machine.
7. Train the caption validation model using one of the following commands:  
    * To train with learned word embeddings:
`python exp-referit/train_cls.py $GPU_ID`
    * To train with GloVe embeddings: `python exp-referit/train_cls_glove.py $GPU_ID`


### Evaluation
8. Select the GPU you want to use during testing: `export GPU_ID=<gpu id>`. Use 0 for `<gpu id>` if you only have one GPU on your machine. Also, you may need to add the repository root directory to Python's module path: `export PYTHONPATH=.:$PYTHONPATH`.
9. Run evaluation for the caption validation model:  
`python coco/exp_test_coco_cls.py $GPU_ID`  .
Look inside the file to see the arguments that can be given to test the model. These arguments should match the ones given to the training batch image file. This should reproduce the results in the paper.
