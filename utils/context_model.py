import torch
from torch import nn
from transformers import RobertaModel, HubertModel, AutoModel, Data2VecAudioModel
from utils.cross_attn_encoder import CMELayer, BertConfig



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# English text model + context
class roberta_en_context(nn.Module):
            
    def __init__(self):        
        super().__init__() 
        self.roberta_model = AutoModel.from_pretrained('roberta-large') # with context, we can improve using a larger model
        self.classifier = nn.Linear(1024*2, 1)    
   
    def forward(self, input_ids, attention_mask, context_input_ids, context_attention_mask):        
        raw_output = self.roberta_model(input_ids, attention_mask, return_dict=True)        
        input_pooler = raw_output["pooler_output"]    # Shape is [batch_size, 1024]

        context_output = self.roberta_model(context_input_ids, context_attention_mask, return_dict=True)
        context_pooler = context_output["pooler_output"]   # Shape is [batch_size, 1024]

        pooler = torch.cat((input_pooler, context_pooler), dim=1)
        output = self.classifier(pooler)                    # Shape is [batch_size, 1]
        return output
    

# English text+audio model + context
class rob_d2v_cc_context(nn.Module):            
    def __init__(self, config):        
        super().__init__()
        self.roberta_model = RobertaModel.from_pretrained('roberta-large')
        self.data2vec_model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base")
        
        self.T_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(1024*2, 1)
           )           
        self.A_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768*2, 1)
          )
        self.fused_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(1024*2+768*2, 1024*2),
            nn.ReLU(),
            nn.Linear(1024*2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )
        
        
    def forward(self, text_inputs, text_mask, text_context_inputs, text_context_mask, audio_inputs, audio_mask, audio_context_inputs, audio_context_mask):
        # text feature extraction
        raw_output = self.roberta_model(text_inputs, text_mask, return_dict=True)        
        input_pooler = raw_output["pooler_output"]    # Shape is [batch_size, 1024]

        # text context feature extraction
        raw_output_context = self.roberta_model(text_context_inputs, text_context_mask, return_dict=True)
        context_pooler = raw_output_context["pooler_output"]    # Shape is [batch_size, 1024]

        # audio feature extraction
        audio_out = self.data2vec_model(audio_inputs, audio_mask, output_attentions=True)
        A_hidden_states = audio_out.last_hidden_state
        ## average over unmasked audio tokens
        A_features = []
        audio_mask_idx_new = []
        for batch in range(A_hidden_states.shape[0]):
            layer = 0
            while layer<12:
                try:
                    padding_idx = sum(audio_out.attentions[layer][batch][0][0]!=0)
                    audio_mask_idx_new.append(padding_idx)
                    break
                except:
                    layer += 1
            truncated_feature = torch.mean(A_hidden_states[batch][:padding_idx],0) #Shape is [768]
            A_features.append(truncated_feature)
        A_features = torch.stack(A_features,0).to(device)   # Shape is [batch_size, 768]
        
        # audio context feature extraction
        audio_context_out = self.data2vec_model(audio_context_inputs, audio_context_mask, output_attentions=True)
        A_context_hidden_states = audio_context_out.last_hidden_state
        ## average over unmasked audio tokens
        A_context_features = []
        audio_context_mask_idx_new = []
        for batch in range(A_context_hidden_states.shape[0]):
            layer = 0
            while layer<12:
                try:
                    padding_idx = sum(audio_context_out.attentions[layer][batch][0][0]!=0)
                    audio_context_mask_idx_new.append(padding_idx)
                    break
                except:
                    layer += 1
            truncated_feature = torch.mean(A_context_hidden_states[batch][:padding_idx],0) #Shape is [768]
            A_context_features.append(truncated_feature)
        A_context_features = torch.stack(A_context_features,0).to(device)   # Shape is [batch_size, 768]

        T_features = torch.cat((input_pooler, context_pooler), dim=1)    # Shape is [batch_size, 1024*2]
        A_features = torch.cat((A_features, A_context_features), dim=1)  # Shape is [batch_size, 768*2]
        T_output = self.T_output_layers(T_features)                    # Shape is [batch_size, 1]
        A_output = self.A_output_layers(A_features)                    # Shape is [batch_size, 1]
        
        fused_features = torch.cat((T_features, A_features), dim=1)    # Shape is [batch_size, 1024*2+768*2]
        fused_output = self.fused_output_layers(fused_features)        # Shape is [batch_size, 1]

        return {
                'T': T_output, 
                'A': A_output, 
                'M': fused_output
        }
