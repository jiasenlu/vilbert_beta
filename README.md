# ViLBERT <img src="fig/vilbert_trim.png" width="45">

Code and pre-trained models for **[ViLBERT: Pretraining Task-Agnostic VisiolinguisticRepresentations for Vision-and-Language Tasks](https://arxiv.org/abs/1908.02265)**.

<span style="color:blue"> *Note: This codebase is still in beta release to replicate the paper's preformance. * </span>

## Repository Setup

1. Create a fresh conda environment, and install all dependencies.

```text
conda create -n vilbert python=3.6
conda activate vilbert
git clone https://github.com/jiasenlu/vilbert_beta
cd vilbert_beta
pip install -r requirements.txt
```

2. Install pytorch
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

3. Install apx, follows https://github.com/NVIDIA/apex

4. compile tools

```
cd tool/refer
make
```
## Data Setup

Check `README.md` under `data` for more details.  Check  `vlbert_tasks.yml` for more details. 


## Pre-trained model for Evaluation

| Model | Objective | Link |
|:-------:|:------:|:------:|
|ViLBERT 6-Layer| Conceptual Caption |[Google Drive](https://drive.google.com/drive/folders/1Re0L75uazH3Qrep_aRgtaVelDEz4HV9c?usp=sharing)|
|ViLBERT 6-Layer| VQA |[Google Drive](https://drive.google.com/drive/folders/1nrcVww0u_vozcFRQVr58-YH5LOU1ZiWT?usp=sharing)|
|ViLBERT 6-Layer| VCR |[Google Drive](https://drive.google.com/drive/folders/1QJuMzBarTKU_hAWDSZm60rWiDnbAVEVZ?usp=sharing)|
|ViLBERT 6-Layer| RefCOCO+ |[Google Drive](https://drive.google.com/drive/folders/1GWY2fEbZCYHkcnxd0oysU0olfPdzcD3l?usp=sharing)|
|ViLBERT 6-Layer| Image Retrieval |[Google Drive](https://drive.google.com/drive/folders/18zUTF3ZyOEuOT1z1aykwtIkBUhfROmJo?usp=sharing)|

## Evaluation

### Zero-Shot Image Retrieval

We can directly use the Pre-trained ViLBERT model for zero-shot image retrieval tasks on Flickr30k. 

1: Download the pretrained model with objective `Conceptual Caption` and put it under `save`

2: Update `featyres_h5path1` and `val_annotations_jsonpath` in  `vlbert_task.yml` to load the Flickr30k testset image feature and jsonfile (defualt is training feature). 

3: Use the following command to evaluate pre-trained 6 layer ViLBERT model. (only support single GPU for evaluation now):

```bash
python eval_retrieval.py --bert_model bert-base-uncased --from_pretrained save/bert_base_6_layer_6_connect/pytorch_model_9.bin --config_file config/bert_base_6layer_6conect.json --task 3 --split test --batch_size 1 --zero_shot
```

### Image Retrieval

1: Download the pretrained model with objective `Image Retrieval` and put it under `save`

2: Update `featyres_h5path1` and `val_annotations_jsonpath` in  `vlbert_task.yml` to load the Flickr30k testset image feature and jsonfile (defualt is training feature). 

3: Use the following command to evaluate pre-trained 6 layer ViLBERT model. (only support single GPU for evaluation now):

```bash
python eval_retrieval.py --bert_model bert-base-uncased --from_pretrained save/RetrievalFlickr30k_bert_base_6layer_6conect-pretrained/pytorch_model_19.bin --config_file config/bert_base_6layer_6conect.json --task 3 --split test --batch_size 1
```

### VQA

1: Download the pretrained model with objective `VQA` and put it under `save`

2: To test on held out validation split, use the following command: 

```
python eval_tasks.py --bert_model bert-base-uncased --from_pretrained save/VQA_bert_base_6layer_6conect-pretrained/pytorch_model_19.bin --config_file config/bert_base_6layer_6conect.json --task 0 --split minval
```

### VCR

1: Download the pretrained model with objective `VCR` and put it under `save`

2: To test on VCR Q->A

```
python eval_tasks.py --bert_model bert-base-uncased --from_pretrained save/VCR_Q-A-VCR_QA-R_bert_base_6layer_6conect-pretrained/pytorch_model_19.bin --config_file config/bert_base_6layer_6conect.json --task 1 --split val
```

3: To test on VCR QA->R

```
python eval_tasks.py --bert_model bert-base-uncased --from_pretrained save/VCR_Q-A-VCR_QA-R_bert_base_6layer_6conect-pretrained/pytorch_model_19.bin --config_file config/bert_base_6layer_6conect.json --task 2 --split val
```

### RefCOCO+

1: Download the pretrained model with objective `RefCOCO+` and put it under `save`

2: We use the Pre-computed detections/masks from [MAttNet](https://github.com/lichengunc/MAttNet) for fully-automatic comprehension task, Check the MAttNet repository for more details. 

3: To test on the RefCOCO+ val set and use the following command:

```bash
python eval_tasks.py --bert_model bert-base-uncased --from_pretrained save/refcoco+_bert_base_6layer_6conect-pretrained/pytorch_model_19.bin --config_file config/bert_base_6layer_6conect.json --task 4
```

## Visiolinguistic Pre-training

Once you extracted all the image features, to train a 6-layer ViLBERT model on conceptual caption:

```bash
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 train_concap.py --from_pretrained bert-base-uncased --bert_model bert-base-uncased --conf
ig_file config/bert_base_6layer_6conect.json --learning_rate 1e-4 --train_batch_size 512 --save_name pretrained
```

### Train ViLBERT for DownStream Tasks

### VQA 

To fintune a 6-layer ViLBERT model for VQA with 8 GPU. `--tasks 0` means VQA tasks. Check `vlbert_tasks.yml` for more settings for VQA tasks.  

```bash
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 train_tasks.py --bert_model bert-base-uncased --from_pretrained save/bert_base_6_layer_6_connect_freeze_0/pytorch_model_8.bin  --config_file config/bert_base_6layer_6conect.json  --learning_rate 4e-5 --num_workers 16 --tasks 0 --save_name pretrained
```

### VCR

Similarly, to finetune a 6-layer vilbert model for VCR task, run the following commands. Here we joint train `Q->A ` and `QA->R` tasks, so the tasks is specified as `--tasks 1-2`

```bash
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 train_tasks.py --bert_model bert-base-uncased --from_pretrained save/bert_base_6_layer_6_connect_freeze_0/pytorch_model_8.bin  --config_file config/bert_base_6layer_6conect.json  --learning_rate 2e-5 --num_workers 16 --tasks 1-2 --save_name pretrained
```

### Image Retrieval

```bash
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 train_tasks.py --bert_model bert-base-uncased --from_pretrained save/bert_base_6_layer_6_connect_freeze_0/pytorch_model_8.bin  --config_file config/bert_base_6layer_6conect.json  --learning_rate 4e-5 --num_workers 9 --tasks 3 --save_name pretrained
```

### Refer Expression

```bash
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 train_tasks.py --bert_model bert-base-uncased --from_pretrained save/bert_base_6_layer_6_connect_freeze_0/pytorch_model_8.bin  --config_file config/bert_base_6layer_6conect.json  --learning_rate 4e-5 --num_workers 16 --tasks 4 --save_name pretrained
```

- For single GPU training, use smaller batch size and simply remove ` -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 ` 

## References

If you find this code is useful for your research, please cite our paper

```
@article{lu2019vilbert,
  title={ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks},
  author={Lu, Jiasen and Batra, Dhruv and Parikh, Devi and Lee, Stefan},
  journal={arXiv preprint arXiv:1908.02265},
  year={2019}
}
```



## Why does ViLBERT look like <img src="fig/vilbert_trim.png" width="45">? 

<p align="center">
<img src="fig/vilbert.png" width="400" >
</p>
