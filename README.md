# Multimodal Multi-Loss Fusion Network (MMML)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-modality-multi-loss-fusion-network/multimodal-sentiment-analysis-on-cmu-mosi)](https://paperswithcode.com/sota/multimodal-sentiment-analysis-on-cmu-mosi?p=multi-modality-multi-loss-fusion-network)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-modality-multi-loss-fusion-network/multimodal-sentiment-analysis-on-mosi)](https://paperswithcode.com/sota/multimodal-sentiment-analysis-on-mosi?p=multi-modality-multi-loss-fusion-network)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-modality-multi-loss-fusion-network/multimodal-sentiment-analysis-on-ch-sims)](https://paperswithcode.com/sota/multimodal-sentiment-analysis-on-ch-sims?p=multi-modality-multi-loss-fusion-network)
	


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/multi-modality-multi-loss-fusion-network/multimodal-sentiment-analysis-on-cmu-mosei-1)](https://paperswithcode.com/sota/multimodal-sentiment-analysis-on-cmu-mosei-1?p=multi-modality-multi-loss-fusion-network)

Paper link: https://arxiv.org/abs/2308.00264 (NAACL2024 oral)


## Updates

**07/19/2024: Add CME for context modeling; upload requirements.txt; skip invalid audio files in dataloader**

#### New Training Results (CME+context):

##### CMU-MOSI

| Seed   | Has0_acc_2 | Has0_F1_score | Non0_acc_2 | Non0_F1_score | Mult_acc_5 | Mult_acc_7 | MAE    | Corr   |
|--------|------------|---------------|------------|---------------|------------|------------|--------|--------|
| 1      | 0.8921     | 0.8918        | 0.9131     | 0.9131        | 0.5991     | 0.5248     | 0.5687 | 0.8803 |
| 11     | 0.8790     | 0.8786        | 0.8979     | 0.8978        | 0.5889     | 0.5160     | 0.5677 | 0.8790 |
| 111    | 0.8834     | 0.8832        | 0.8994     | 0.8995        | 0.6122     | 0.5408     | 0.5356 | 0.8878 |
|        |            |               |            |               |            |            |        |        |
| Avg    | 0.8848     | 0.8845        | 0.9035     | 0.9035        | 0.6001     | 0.5272     | 0.5573 | 0.8824 |

##### CMU-MOSEI

| Seed   | Has0_acc_2 | Has0_F1_score | Non0_acc_2 | Non0_F1_score | Mult_acc_5 | Mult_acc_7 | MAE    | Corr   |
|--------|------------|---------------|------------|---------------|------------|------------|--------|--------|
| 1      | 0.8676     | 0.8674        | 0.8822     | 0.8800        | 0.5864     | 0.5660     | 0.4836 | 0.8196 |
| 11     | 0.8669     | 0.8673        | 0.8828     | 0.8810        | 0.5787     | 0.5563     | 0.4986 | 0.8085 |
| 111    | 0.8562     | 0.8578        | 0.8817     | 0.8803        | 0.5585     | 0.5209     | 0.5393 | 0.8146 |
|        |            |               |            |               |            |            |        |        |
| Avg    | 0.8636     | 0.8642        | 0.8822     | 0.8804        | 0.5745     | 0.5477     | 0.5072 | 0.8142 |

#### Settings
GPU: NVIDIA RTX A6000

Parameter: Default (batch_size=8, lr=5e-6, loss='MTA', text_context_len=2, audio_context_len=1, num_hidden_layers=5)

Details: We use the large version of pre-trained models for this setting. We found that while the large models didn't improve upon the base models in context-free scenarios, they boosted improvement when using context. 


## Environment setup
1. create a new environment using conda or pip (We use Python 3.8.10)
2. ```pip install -r requirements.txt```


## Download Data
The three datasets (CMU-MOSI, CMU-MOSEI, and CH-SIMS) are available from this link: https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk

## Data Directory
To run our preprocessing and training codes directly, please put the necessary files from downloaded data in separate folders as described below.

```
/data/
    mosi/
        raw/
        label.csv
    mosei/
        raw/
        label.csv
    sims/
        raw/
        label.csv
```

## Audio Extraction
Before running the training code, please extract the audio from the raw video files.

```
python extract_audio.py

options:
  --dataset DATASET     dataset name (mosi, mosei, or sims) (default: mosi)
```

## Train only with audio features
```
python run_audio.py  

options (optional):
  --lr LR               learning rate (default: 1e-4)
  
  --dataset DATASET     dataset name (mosi, mosei, or sims) (default: sims)
  
  --seed SEED           random seed (default: 1)
  
  --feature FEATURE     feature type: mel spectrogram (spec), smile, or raw (default: spec)
  
  --batch_size BATCH_SIZE
                        batch size (default: 16)
                        
  --early_stop EARLY_STOP
                        early stop (default: 8)
```

## Train with both text and audio features
```
python run.py

options (optional):
  --seed SEED           random seed (default: 1)
  
  --batch_size BATCH_SIZE
                        batch size (default: 8)
                        
  --lr LR               learning rate (default: 5e-6, recommended: 5e-6 for mosi, mosei, 1e-5 for sims)
  
  --model MODEL         concatenate(cc) or cross-modality encoder(cme) (default: cme)
  
  --cme_version VERSION
                        different variations of the fusion network
                        (v1: employs only the fused features, v2: merges the original signal with the fused signal, v3: uses a transformer to combine these two signals.) (default: v1)
                        
  --dataset DATASET     dataset name (mosi, mosei, or sims) (default: mosi)
  
  --num_hidden_layers NUM_HIDDEN_LAYERS
                        number of hidden layers for cross-modality encoder (default: 5)
                        
  --loss LOSS          use multiple losses to train: M: multi-modal, T: text, A: audio
                       (if T -> train only the text subnet; if M -> use the loss from fusion network to train the whole model)
                       (default: MTA)
                       
  --context CONTEXT    weather incorporating context or not (default: True)
  
  --text_context_len LENGTH  the length of the context window for text features (default: 2)
  
  --audio_context_len LENGTH the length of the context window for audio features (default: 1)
```

## Side Note
Please feel free to adopt the code for any combinations of pre-trained models.
