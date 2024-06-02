# Multi-Modality Multi-Loss Fusion Network (MMML)

Paper link: https://arxiv.org/abs/2308.00264

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
                        batch size (default: 16)
                        
  --lr LR               learning rate (default: 5e-6, recommended: 5e-6 for mosi, mosei, 1e-5 for sims)
  
  --model MODEL         concatenate(cc) or cross-modality encoder(cme) (default: cc)
  
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
Please feel free to adopt the code for any combinations of pre-trained models. We have found whisper models to be better audio feature extractors than wave2vec-based model in other projects.