### Download Conceptual Caption
Head to `tools/DownloadConcptualCaption` to download the conceptual caption dataset. 

### Feature Extraction
We use [bottom-up-attention](https://github.com/jiasenlu/bottom-up-attention) to extract detection features from the image. Please follow the instructions to setup the feature extraction pipeline (You can also use detectron instead, but those feature won't work with the pre-trained model.)

I've uploaded the script I use to extract visual feature for ViLBERT, please check [this](https://github.com/jiasenlu/bottom-up-attention/blob/master/tools/generate_tsv.py) script and [this](https://github.com/jiasenlu/bottom-up-attention/blob/master/tools/generate_tsv_gt.py) script to extract feature from image and ground truth bounding box. For example, to extract features from image: 

```bash
python ./tools/generate_tsv.py --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt --out feature/VCR/VCR_resnet101_faster_rcnn_genome.tsv --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --total_group 1 --group_id 0 --split VCR
```

To extract features from the ground truth bounding box:

``` bash
python ./tools/generate_tsv_gt.py --cfg experiments/cfgs/faster_rcnn_end2end_resnet.yml --def models/vg/ResNet-101/faster_rcnn_end2end_final/test_gt.prototxt --out feature/VCR/VCR_gt_resnet101_faster_rcnn_genome.tsv --net data/faster_rcnn_models/resnet101_faster_rcnn_final.caffemodel --total_group 1 --group_id 0 --split VCR_gt
```

This will generate `xxx.tsv` file under `bottom-up-attention/feature`, next we need to convert them into lmdb file. 

For **Conceptual Caption**, we use [TensorPack](https://github.com/tensorpack/tensorpack) to sequentially load the data and do local snuffling from the pool, consider run 
```scripts/conceptual_caption_preprocess_sequential_train.py```
to convert the feature.  

If you want to train the model with distributed training, you need to segment the data into multiple pieces, Since Tensorpack sequential reading didn't have a distributed version. Consider use 

```
scripts/conceptual_caption_preprocess_sequential_train_segment.py
```

to generate the segments. 

For **Downstream Tasks**:

we use the default pytorch dataloader to load the Downstream tasks, consider use the following script to convert the `tsv` file to `lmdb`. 

```bash
scripts/convert_lmdb_VCR.py
scripts/convert_lmdb_VCR_gt.py
```

other tasks follows the same pipeline to extract the image feature since we use the same API to load image features. 

### Download the Feature

You can directly download the feature and test the pre-trained model. (Thanks Jianwei to Hold these files!)

|Task    |                             Link                             |
| :-------------: | :----------------------------------------------------------: |
|       COCO       |  [Dropbox](https://www.dropbox.com/sh/09182lupkagw1ov/AACShSEVClAh6CzbhyIKxmtga?dl=0)|
|       VCR       | [Dropbox](https://www.dropbox.com/sh/9pgxc3njd3iq03o/AADXgnT1HmEdrds7aujTncBGa?dl=0) |
|    RefCOCO+     | [Dropbox](https://www.dropbox.com/sh/4jqadcfkai68yoe/AADHI6dKviFcraeCMdjiaDENa?dl=0)|
| Flickr30k | [Dropbox](https://www.dropbox.com/sh/qqk1xlhkqjyek8q/AAADni5hVBV2PAC8R_13xpIja?dl=0)|


### Meta Data
We also provide the Meta data for ViLBERT, you can download them and run the code more eailiy. 

[Google drive](https://drive.google.com/drive/folders/1o7sCLl1_PKCoaGvigCr_uGuBg6koOJm8?usp=sharing)
