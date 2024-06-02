import pandas as pd
import torch
import opensmile
import numpy as np
import torchaudio
from torchaudio import transforms
import math, random
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from transformers import Wav2Vec2FeatureExtractor

# global variables
# spectrogram
N_MELS = 128
N_FFT = 960 #960/16000 = 0.06
HOP_Length = 320 #320/16000 = 0.02
MAX_DUR = 6000
#openSMILE
MAX_FEA_LEN = 300 #6000ms



# AudioUtil class adapted from https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5
class AudioUtil():
    
    # ----------------------------
    # Load an audio file. Return the signal as a tensor and the sample rate
    # ----------------------------
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)

    # ----------------------------
    # Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
    # ----------------------------
    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr//1000 * max_ms

        if (sig_len > max_len):
            # Truncate the signal to the given length
            sig = sig[:,:max_len]

        elif (sig_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)

        return (sig, sr)
    
    # ----------------------------
    # Convert the given audio to the desired number of channels
    # ----------------------------
    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud

        if (sig.shape[0] == new_channel):
            # Nothing to do
            return aud

        if (new_channel == 1):
            # Convert from stereo to mono by selecting only the first channel
            resig = sig[:1, :]
        else:
            # Convert from mono to stereo by duplicating the first channel
            resig = torch.cat([sig, sig])

        return ((resig, sr))
    
    # ----------------------------
    # Generate a Spectrogram
    # ----------------------------
    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig,sr = aud
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)

    
def stanParameters(dataset, cv):
    # ----------------------------
    # extract standadization parameters from training set only
    #  dataset: sims, mosi
    #  cv: True or False (spectrograms or openSMILE)
    # ----------------------------
    
    if dataset == 'sims':
        csv_path = 'data/SIMS/label.csv'
        audio_directory = "data/SIMS/wav"
    elif dataset == "mosi":
        csv_path = 'data/MOSI/label.csv'
        audio_directory = "data/MOSI/WAV_16000/Segmented"
        
    # open csv file    
    df = pd.read_csv(csv_path)
    df = df[df['mode']=='train'].reset_index()
    
    # store audio path
    audio_file_paths = []
    #loop through the csv entries to record path
    if dataset == 'sims':
        for i in range(0,len(df)):
            clip_id = str(df['clip_id'][i])
            for j in range(4-len(clip_id)):
                clip_id = '0'+clip_id
            file_name = str(df['video_id'][i]) + '/' + clip_id + '.wav'      
            path = audio_directory + "/" + file_name
            audio_file_paths.append(path)
    elif dataset == 'mosi':
        for i in range(0,len(df)):
            file_name = str(df['video_id'][i])+'_'+str(df['clip_id'][i])+'.wav'      
            path = audio_directory + "/" + file_name
            audio_file_paths.append(path)
    
    if cv: # spectrogram
        # standardize spectrograms separately for the two channels of audio
        TrainFeatures_0, TrainFeatures_1 = [],[] 
        for i,path in enumerate(audio_file_paths):
            # extract features
            aud = AudioUtil.open(path)
            rechan = AudioUtil.rechannel(aud, 2)
            dur_aud = AudioUtil.pad_trunc(rechan, MAX_DUR)
            features = AudioUtil.spectro_gram(dur_aud, n_mels=N_MELS, n_fft=N_FFT, hop_len=None)
            if i==0:
                TrainFeatures_0 = torch.flatten(features[0])
                TrainFeatures_1 = torch.flatten(features[1])
            else:
                TrainFeatures_0 = torch.concatenate((TrainFeatures_0,torch.flatten(features[0])))
                TrainFeatures_1 = torch.concatenate((TrainFeatures_1,torch.flatten(features[1])))
        # define standard scaler
        scaler0 = StandardScaler()
        scaler1 = StandardScaler()
        # transform data
        scaled_0 = scaler0.fit_transform(TrainFeatures_0.reshape(-1, 1))
        scaled_1 = scaler1.fit_transform(TrainFeatures_1.reshape(-1, 1))
        return scaler0.mean_, scaler0.var_, scaler1.mean_, scaler1.var_
    else: # openSMILE
        # feature extrator
        smile_lld = opensmile.Smile(
                    feature_set=opensmile.FeatureSet.ComParE_2016,
                    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
                )
        smile_delta = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors_Deltas,
        ) 

        # extracting
        TrainFeatures = [], 
        for i,path in enumerate(audio_file_paths):
            # extract features  
            lld = smile_lld.process_file(path)       
            delta = smile_delta.process_file(path)

            # concatenate features
            features = np.concatenate((lld.to_numpy(), delta.to_numpy()[:-2]), axis=1)

            # truncate features
            if features.shape[0]>MAX_FEA_LEN:
                features = features[:MAX_FEA_LEN] 

            if i==0:
                TrainFeatures = features
            else:
                TrainFeatures = np.concatenate((TrainFeatures,features), axis=0)

        # define standard scaler
        scaler = StandardScaler()
        # transform data
        scaled = scaler.fit_transform(TrainFeatures)  
        return scaler.mean_, scaler.var_ 
    

            
class AudioDataset(torch.utils.data.Dataset):    
    def __init__(self, dataset, mode, feature):
        # ----------------------------
        # Argument List:
        #  dataset: sims, mosi
        #  mode: train, test, valid
        #  feature: spec, smile, raw (spectrograms or openSMILE or raw audio for pre-trained model)

        # ----------------------------
        self.feature = feature
        
        if dataset == 'sims':
            csv_path = 'data/SIMS/label.csv'
            audio_directory = "data/SIMS/wav"
        elif dataset == "mosi":
            csv_path = 'data/MOSI/label.csv'
            audio_directory = "data/MOSI/wav"
            
        # open csv file    
        df = pd.read_csv(csv_path)
        df = df[df['mode']==mode].reset_index()
        
        # store labels
        if dataset == 'sims':
            self.targets = df['label_A']
        else:
            self.targets = df['label']
            
        # store audio path
        self.audio_file_paths = []    
        #loop through the csv entries to record path
        if dataset == 'sims':
            for i in range(0,len(df)):
                clip_id = str(df['clip_id'][i])
                for j in range(4-len(clip_id)):
                    clip_id = '0'+clip_id
                file_name = str(df['video_id'][i]) + '/' + clip_id + '.wav'
                file_path = audio_directory + "/" + file_name
                self.audio_file_paths.append(file_path)
             
        elif dataset == 'mosi':
            for i in range(0,len(df)):
                file_name = str(df['video_id'][i])+'/'+str(df['clip_id'][i])+'.wav'
                file_path = audio_directory + "/" + file_name
                self.audio_file_paths.append(file_path)
        
        # initilize feature extractor or standardization parameters
        if feature == 'smile': 
            # initialize openSMILE
            self.smile_lld = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
            )
            self.smile_delta = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.LowLevelDescriptors_Deltas,
            )
            # initialize standardize parameters (calculate using the stanParameters() function)
            if dataset == 'sims':
                with open('data/SIMS/simsOpenMean.npy', 'rb') as f:
                    self.mean = np.load(f)
                with open('data/SIMS/simsOpenVar.npy', 'rb') as f:
                    self.var = np.load(f)
            else:
                with open('data/MOSI/mosiOpenMean.npy', 'rb') as f:
                    self.mean = np.load(f)
                with open('data/MOSI/mosiOpenVar.npy', 'rb') as f:
                    self.var = np.load(f) 
        elif feature == 'spec':
            if dataset == 'mosi': # calculate using the stanParameters() function
                self.MEAN_0 = -34.57500023
                self.VAR_0 = 492.96437826
                self.MEAN_1 = -34.62100053
                self.VAR_1 = 491.16052842
            else:
                self.MEAN_0 = -30.24599383
                self.VAR_0 = 473.88704116
                self.MEAN_1 = -30.35495954
                self.VAR_1 = 472.0672438
        else:
            self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
                
    def __len__(self):
        return len(self.targets)
            
    def __getitem__(self, index):
        if self.feature == 'smile':
            # extract features  
            lld = self.smile_lld.process_file(self.audio_file_paths[index])       
            delta = self.smile_delta.process_file(self.audio_file_paths[index])

            # concatenate features
            features = np.concatenate((lld.to_numpy(), delta.to_numpy()[:-2]), axis=1)

            # truncate features
            if features.shape[0]>MAX_FEA_LEN:
                features = features[:MAX_FEA_LEN]

            # normalize features
            features = (features-self.mean)/(np.sqrt(self.var))
                
            
        elif self.feature == 'spec':
            aud = AudioUtil.open(self.audio_file_paths[index])
            rechan = AudioUtil.rechannel(aud, 2)
            dur_aud = AudioUtil.pad_trunc(rechan, MAX_DUR)
            features = AudioUtil.spectro_gram(dur_aud, n_mels=N_MELS, n_fft=N_FFT, hop_len=HOP_Length)
            features[0] = (features[0]-self.MEAN_0)/np.sqrt(self.VAR_0)
            features[1] = (features[1]-self.MEAN_1)/np.sqrt(self.VAR_1)
            features = features.type(torch.float32)
        else:
            sound,_ = torchaudio.load(self.audio_file_paths[index])
            soundData = torch.mean(sound, dim=0, keepdim=False)
            # audio_features = self.feature_extractor(soundData, sampling_rate=16000, max_length=96000,return_attention_mask=True,truncation=True, padding="max_length")
            # return {
            #     "audio_features": torch.tensor(np.array(audio_features['input_values']), dtype=torch.float).squeeze(),
            #     "masks": torch.tensor(np.array(audio_features['attention_mask']), dtype=torch.long).squeeze(),
            #     "targets": torch.tensor(self.targets[index], dtype=torch.float32)
            # }
            return {
                "audio_features": soundData,
                "targets": torch.tensor(self.targets[index], dtype=torch.float32)
            }
            
        return { 
                "audio_features": features,
                "targets": torch.tensor(self.targets[index], dtype=torch.float32)
        }

    
def collate_fn_smile(batch):
    # ----------------------------
    # dynamic padding collate function for openSMILE feature
    # ----------------------------
    
    # collate
    inputs = []
    targets = []
    length = []
    for i in range(len(batch)):
        inputs.append(torch.from_numpy(batch[i]['audio_features']))
        targets.append(batch[i]['targets'])
        length.append(batch[i]['audio_features'].shape[0])
        
    # dynamic padding
    inputs = pad_sequence(inputs, batch_first=True)
    masks = torch.zeros(len(batch),inputs.shape[1])
    for i in range(len(length)):
        masks[i][length[i]:] = 1
    return {"audio_features": inputs.type(torch.float32),
            "masks": masks,
            "targets": torch.tensor(targets, dtype=torch.float32),
            }  


def collate_fn_raw(batch):
    # collate
    inputs = []
    targets = []
    length = []
    for i in range(len(batch)):
        inputs.append(batch[i]['audio_features'].tolist())
        targets.append(batch[i]['targets'])
        length.append(batch[i]['audio_features'].shape[0])
        
    # dynamic padding
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    audio_features = feature_extractor(inputs, sampling_rate=16000, max_length=96000, return_attention_mask=True,truncation=True, padding='longest')
    
    return {"audio_features": torch.tensor(np.array(audio_features['input_values']), dtype=torch.float),
            "masks": torch.tensor(np.array(audio_features['attention_mask']), dtype=torch.long),
            "targets": torch.tensor(targets, dtype=torch.float32)
            }  



def AudioDataloader(batch_size, dataset, feature):
    # ----------------------------
    # Argument List:
    #  batch size: smaller if using raw features
    #  dataset: sims, mosi
    #  feature: spec, smile, raw (spectrograms or openSMILE or raw audio for pre-trained model)
    # ----------------------------
    train_data = AudioDataset(dataset, 'train', feature)
    test_data = AudioDataset(dataset, 'test', feature)
    val_data = AudioDataset(dataset, 'valid', feature)
 
    if feature == 'smile':
        trainLoader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn_smile, shuffle=True)
        testLoader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn_smile, shuffle=False)
        valLoader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_fn_smile, shuffle=False)
    elif feature == 'raw':
        trainLoader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn_raw, shuffle=True)
        testLoader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn_raw, shuffle=False)
        valLoader = DataLoader(val_data, batch_size=batch_size, collate_fn=collate_fn_raw, shuffle=False)
    else:
        trainLoader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        testLoader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        valLoader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return trainLoader, testLoader, valLoader
