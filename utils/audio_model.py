import torch.nn.functional as F
from torch.nn import init
import torch
from torch import nn
from positional_encodings.torch_encodings import PositionalEncodingPermute1D, Summer
from transformers import Data2VecAudioModel, AutoModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Audio Classification Model
# ----------------------------
class AudioClassifier(nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight)
        self.conv1.bias.data.zero_()
        # conv_layers += [self.conv1, self.relu1, self.bn1]
        conv_layers1 = [self.conv1, self.relu1, self.bn1]
        self.conv1 = nn.Sequential(*conv_layers1)
        
        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight)
        self.conv2.bias.data.zero_()
        # conv_layers += [self.conv2, self.relu2, self.bn2]
        conv_layers2 = [self.conv2, self.relu2, self.bn2]
        self.conv2 = nn.Sequential(*conv_layers2)
        
        # Third Convolution Block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight)
        self.conv3.bias.data.zero_()
        # conv_layers += [self.conv3, self.relu3, self.bn3]
        conv_layers3 = [self.conv3, self.relu3, self.bn3]
        self.conv3 = nn.Sequential(*conv_layers3)
        
        # Fourth Convolution Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight)
        self.conv4.bias.data.zero_()
        # conv_layers += [self.conv4, self.relu4, self.bn4]
        conv_layers4 = [self.conv4, self.relu4, self.bn4]
        self.conv4 = nn.Sequential(*conv_layers4)
        
        # Fifth Convolution Block
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu5 = nn.ReLU()
        self.bn5 = nn.BatchNorm2d(128)
        init.kaiming_normal_(self.conv5.weight)
        self.conv5.bias.data.zero_()
        # conv_layers += [self.conv5, self.relu5, self.bn5]
        conv_layers5 = [self.conv5, self.relu5, self.bn5]
        self.conv5 = nn.Sequential(*conv_layers5)
        
        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier  = nn.Sequential(
                                  # nn.Linear(5120, 512),
                                  # nn.ReLU(),
                                  # nn.Linear(512, 128),
                                  nn.ReLU(),
                                  nn.Linear(128, 1)  
                                  )
 
    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        x = torch.flatten(x, start_dim=1)

        # Linear layer
        out = self.classifier(x)

        # Final output
        return out, x
    

class AudioTransformers(nn.Module):
    def __init__(self, embedding_dim=130):        
        super().__init__() 
        
        self.cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=embedding_dim)
        # self.pos_enc = Summer(PositionalEncoding1D(embedding_dim))
        self.embedding_dim = embedding_dim
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=2, batch_first=True, dim_feedforward=512)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3,enable_nested_tensor=False)
        self.classifier = nn.Linear(embedding_dim, 1)
        
        
    def prepend_cls(self, inputs, masks):
        index = torch.LongTensor([0]).to(device=inputs.device)
        cls_emb = self.cls_emb(index)
        cls_emb = cls_emb.expand(inputs.size(0), 1, self.embedding_dim)
        outputs = torch.cat((cls_emb, inputs), dim=1)
        
        cls_mask = torch.zeros(inputs.size(0), 1).to(device=inputs.device)
        masks = torch.cat((cls_mask, masks), dim=1)
        return outputs, masks
    
    def forward(self, inputs, attention_mask, use_cls=True):
        # transformer to model time relationship
        if use_cls:
            inputs, attention_mask = self.prepend_cls(inputs,attention_mask) # add cls token
            pos_enc = Summer(PositionalEncodingPermute1D(inputs.shape[1]))
            inputs = pos_enc(inputs) # position encoding
            out = self.transformer_encoder(inputs, src_key_padding_mask=attention_mask.bool()) # Shape is [batch_size, seq_length, 130]
            hidden_states = out[:,0,:]   # Shape is [batch_size, 130]
        else:
            pos_enc = Summer(PositionalEncodingPermute1D(inputs.shape[1]))
            inputs = pos_enc(inputs)
            out = self.transformer_encoder(inputs, src_key_padding_mask=attention_mask) # Shape is [batch_size, seq_length, 130]
            hidden_states = torch.mean(out, dim=1, keepdim=False) # Shape is [batch_size, 130]
        
        # classify        
        output = self.classifier(hidden_states)                    # Shape is [batch_size, 1]
        return output, out[:,1:,:]
    

# mandarin
class hubert(nn.Module):            
    def __init__(self):        
        super().__init__()
        self.hubert_model = AutoModel.from_pretrained('TencentGameMate/chinese-hubert-base')
        self.classifier = nn.Linear(768, 1)           
   
    def forward(self, input_ids, attention_mask):        
        raw_output = self.hubert_model(input_ids, attention_mask, output_attentions=True)        
        hidden_states = raw_output.last_hidden_state    # Shape is [batch_size, seq_length, 768]
        
        # average hidden states using attn_mask for each sample in a batch
        features = []
        for batch in range(hidden_states.shape[0]):
            # find a layer that has attentions available
            layer = 0
            while layer<12:
                try:
                    padding_idx = sum(raw_output.attentions[layer][batch][0][0]!=0)
                    break
                except:
                    layer += 1
            
            avg_feature = torch.mean(hidden_states[batch][:padding_idx],0) #Shape is [768]
            features.append(avg_feature)
            
        features = torch.stack(features,0).to(device) #Shape is [batch_size, 768]
        
        output = self.classifier(features)                    # Shape is [batch_size, 1]
        
        return output, features
    

# English
class data2vec(nn.Module):            
    def __init__(self):        
        super().__init__() 
        self.data2vec_model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base")      
        self.classifier = nn.Linear(768, 1)           
   
    def forward(self, input_ids, attention_mask):        
        raw_output = self.data2vec_model(input_ids, return_dict=True)        
        pooler = raw_output.last_hidden_state    # Shape is [batch_size, seq_length, 768]
        pooler = torch.mean(pooler, dim=1, keepdim=False) # Shape is [batch_size, 768]
        
        output = self.classifier(pooler)                    # Shape is [batch_size, 2]

        return output, pooler