import torch
from torch import nn
import transformers
import torchaudio
from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np
import string


class Dataset_sims(torch.utils.data.Dataset):
    # Argument List
    #  csv_path: path to the csv file
    #  audio_directory: path to the audio files
    #  mode: train, test, valid
    
    def __init__(self, csv_path, audio_directory, mode):       
        df = pd.read_csv(csv_path)
        df = df[df['mode']==mode].reset_index()
        
        # store labels
        self.targets_M = df['label']
        self.targets_T = df['label_T']
        self.targets_A = df['label_A']
        
        # store texts
        self.texts = df['text']
        self.tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        
        # store audio
        self.audio_file_paths = []

        for i in range(0,len(df)):
            clip_id = str(df['clip_id'][i])
            for j in range(4-len(clip_id)):
                clip_id = '0'+clip_id
            file_name = str(df['video_id'][i]) + '/' + clip_id + '.wav'
            file_path = audio_directory + "/" + file_name
            self.audio_file_paths.append(file_path)
      
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)   
        
        
    def __getitem__(self, index):
       # extract text features
        text = str(self.texts[index])         
        tokenized_text = self.tokenizer(
            text,            
            max_length = 64,                                
            padding = "max_length",     # Pad to the specified max_length. 
            truncation = True,          # Truncate to the specified max_length. 
            add_special_tokens = True,  # Whether to insert [CLS], [SEP], <s>, etc.   
            return_attention_mask = True            
        )               
                
        # extract audio features    
        sound,_ = torchaudio.load(self.audio_file_paths[index])
        soundData = torch.mean(sound, dim=0, keepdim=False)
        features = self.feature_extractor(soundData, sampling_rate=16000, max_length=96000,return_attention_mask=True,truncation=True, padding="max_length")
        audio_features = torch.tensor(np.array(features['input_values']), dtype=torch.float32).squeeze()
        audio_masks = torch.tensor(np.array(features['attention_mask']), dtype=torch.long).squeeze()
            
        return { # text
                "text_tokens": tokenized_text["input_ids"],
                "text_masks": tokenized_text["attention_mask"],
                 # audio
                "audio_inputs": audio_features,
                "audio_masks": audio_masks,
                 # labels
                "target": {
                    "M": self.targets_M[index],
                    "T": self.targets_T[index],
                    "A": self.targets_A[index]
                }
                }
    
    def __len__(self):
        return len(self.targets_M)



class Dataset_mosi(torch.utils.data.Dataset):
    # Argument List
    #  csv_path: path to the csv file
    #  audio_directory: path to the audio files
    #  mode: train, test, valid
    #  text_context_length
    #  audio_context_length
    
    def __init__(self, csv_path, audio_directory, mode, text_context_length=2, audio_context_length=1):
        df = pd.read_csv(csv_path)
        df = df[df['mode']==mode].sort_values(by=['video_id','clip_id']).reset_index()
        
        # store labels
        self.targets_M = df['label']
        
        # store texts
        df['text'] = df['text'].str[0]+df['text'].str[1::].apply(lambda x: x.lower())
        self.texts = df['text']
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")

        # store audio
        self.audio_file_paths = []
        ## loop through the csv entries
        for i in range(0,len(df)):
            file_name = str(df['video_id'][i])+'/'+str(df['clip_id'][i])+'.wav'
            file_path = audio_directory + "/" + file_name
            self.audio_file_paths.append(file_path)
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)

        # store context
        self.video_id = df['video_id']
        self.text_context_length = text_context_length
        self.audio_context_length = audio_context_length
        
    def __getitem__(self, index):
        # load text
        text = str(self.texts[index])             

        # load text context
        text_context = ''
        for i in range(1, self.text_context_length+1):
            if index - i < 0 or self.video_id[index] != self.video_id[index - i]:
                break
            else:
                context = str(self.texts[index - i])
                text_context = context + '</s>' + text_context
        
        # tokenize text
        tokenized_text = self.tokenizer(
                text,            
                max_length = 96,                                
                padding = "max_length",     # Pad to the specified max_length. 
                truncation = True,          # Truncate to the specified max_length. 
                add_special_tokens = True,  # Whether to insert [CLS], [SEP], <s>, etc.   
                return_attention_mask = True            
            )  
        
        # tokenize text context
        text_context = text_context[:-4]
        tokenized_context = self.tokenizer(
            text_context,            
            max_length = 96,                                
            padding = "max_length",     # Pad to the specified max_length. 
            truncation = True,          # Truncate to the specified max_length. 
            add_special_tokens = True,  # Whether to insert [CLS], [SEP], <s>, etc.   
            return_attention_mask = True            
        )

        # load audio
        sound,_ = torchaudio.load(self.audio_file_paths[index])
        soundData = torch.mean(sound, dim=0, keepdim=False)

        # load audio context
        audio_context = torch.tensor([])
        for i in range(1, self.audio_context_length+1):
            if index - i < 0 or self.video_id[index] != self.video_id[index - i]:
                break
            else:
                context,_ = torchaudio.load(self.audio_file_paths[index - i])
                contextData = torch.mean(context, dim=0, keepdim=False)
                audio_context = torch.cat((contextData, audio_context), 0)

        # extract audio features
        features = self.feature_extractor(soundData, sampling_rate=16000, max_length=96000,return_attention_mask=True,truncation=True, padding="max_length")
        audio_features = torch.tensor(np.array(features['input_values']), dtype=torch.float32).squeeze()
        audio_masks = torch.tensor(np.array(features['attention_mask']), dtype=torch.long).squeeze()

        # extract audio context features
        if len(audio_context) == 0:
            audio_context_features = torch.zeros(96000)
            audio_context_masks = torch.zeros(96000)
        else:
            features = self.feature_extractor(audio_context, sampling_rate=16000, max_length=96000,return_attention_mask=True,truncation=True, padding="max_length")
            audio_context_features = torch.tensor(np.array(features['input_values']), dtype=torch.float32).squeeze()
            audio_context_masks = torch.tensor(np.array(features['attention_mask']), dtype=torch.long).squeeze()

        return { # text
                "text_tokens": torch.tensor(tokenized_text["input_ids"], dtype=torch.long),
                "text_masks": torch.tensor(tokenized_text["attention_mask"], dtype=torch.long),
                "text_context_tokens": torch.tensor(tokenized_context["input_ids"], dtype=torch.long),
                "text_context_masks": torch.tensor(tokenized_context["attention_mask"], dtype=torch.long),
                # audio
                "audio_inputs": audio_features,
                "audio_masks": audio_masks,
                "audio_context_inputs": audio_context_features,
                "audio_context_masks": audio_context_masks,
                 # labels
                "targets": torch.tensor(self.targets_M[index], dtype=torch.float),
                }
    
    def __len__(self):
        return len(self.targets_M)
    
    
def collate_fn_sims(batch):   
    text_tokens = []  
    text_masks = []
    audio_inputs = []  
    audio_masks = []
    
    targets_M = []
    targets_T = []
    targets_A = []
   
    # organize batch
    for i in range(len(batch)):
        # text
        text_tokens.append(batch[i]['text_tokens'])
        text_masks.append(batch[i]['text_masks'])
        #audio
        audio_inputs.append(batch[i]['audio_inputs'])
        audio_masks.append(batch[i]['audio_masks'])

       # labels
        targets_M.append(batch[i]['target']['M'])
        targets_T.append(batch[i]['target']['T'])
        targets_A.append(batch[i]['target']['A'])        
       
    return {
            # text
            "text_tokens": torch.tensor(text_tokens, dtype=torch.long),
            "text_masks": torch.tensor(text_masks, dtype=torch.long),           
            # audio
            "audio_inputs": torch.stack(audio_inputs),
            "audio_masks": torch.stack(audio_masks),
            # labels
            "targets": {
                    "M": torch.tensor(targets_M, dtype=torch.float32),
                    "T": torch.tensor(targets_T, dtype=torch.float32),
                    "A": torch.tensor(targets_A, dtype=torch.float32)
                }
            }   


def data_loader(batch_size, dataset, text_context_length=2, audio_context_length=1):
    if dataset == 'mosi':
        csv_path = 'data/MOSI/label.csv'
        audio_file_path = "data/MOSI/wav"
        train_data = Dataset_mosi(csv_path, audio_file_path, 'train', text_context_length=text_context_length, audio_context_length=audio_context_length)
        test_data = Dataset_mosi(csv_path, audio_file_path, 'test', text_context_length=text_context_length, audio_context_length=audio_context_length)
        val_data = Dataset_mosi(csv_path, audio_file_path, 'valid', text_context_length=text_context_length, audio_context_length=audio_context_length)
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, val_loader
    elif dataset == 'mosei':
        csv_path = 'data/MOSEI/label.csv'
        audio_file_path = "data/MOSEI/wav"
        train_data = Dataset_mosi(csv_path, audio_file_path, 'train', text_context_length=text_context_length, audio_context_length=audio_context_length)
        test_data = Dataset_mosi(csv_path, audio_file_path, 'test', text_context_length=text_context_length, audio_context_length=audio_context_length)
        val_data = Dataset_mosi(csv_path, audio_file_path, 'valid', text_context_length=text_context_length, audio_context_length=audio_context_length)
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, val_loader
    else:
        csv_path = 'data/SIMS/label.csv'
        audio_file_path = "data/SIMS/wav"
        train_data = Dataset_sims(csv_path, audio_file_path, 'train')
        test_data = Dataset_sims(csv_path, audio_file_path, 'test')
        val_data = Dataset_sims(csv_path, audio_file_path, 'valid')
        
        train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn_sims, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn_sims, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_fn_sims, shuffle=False)
        return train_loader, test_loader, val_loader
