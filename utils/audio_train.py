import torch
from torch import nn
from tqdm import tqdm
from utils.audio_loader import AudioDataloader
from utils.audio_model import AudioClassifier, AudioTransformers, data2vec, hubert
from utils.metricsTop import MetricsTop
import random
import numpy as np

# global variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " %(key, src_dict[key]) 
    return dst_str

class modelConfig(object):
    """Configuration class to store the configurations of training.
    """
    def __init__(self,
                train_mode = 'regression',
                 model_save_path = 'checkpoint/',
                 learning_rate = 1e-4,
                 dataset_name = 'sims',
                 seed = 111,
                 model_name = 'custom',
                 feature = 'spec', # spec or smile or raw
                 batch_size = 16,
                 early_stop = 8
                ):

        self.train_mode = train_mode
        self.learning_rate = learning_rate
        self.dataset = dataset_name
        self.seed = seed
        self.model_name = model_name
        self.model_save_path = model_save_path
        self.feature = feature
        self.batch_size = batch_size
        self.early_stop = early_stop
        
class audio_trainer():
    def __init__(self, config):

        self.config = config
        self.criterion = nn.MSELoss() if config.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(config.train_mode).getMetics(config.dataset)
        
    def train(self, model, data_loader):    
        
        model.train()                                # Put the model in training mode.        
        optimizer = torch.optim.AdamW(model.parameters(),
            lr=self.config.learning_rate
        )
        train_loss = 0         

        for batch in tqdm(data_loader):                    # Loop over all batches.
            if self.config.feature == 'spec':
                inputs = batch["audio_features"].to(device)
                targets = batch["targets"].to(device).view(-1, 1) 

                optimizer.zero_grad()                    # To zero out the gradients.

                output,_ = model(inputs)  # Predictions from 1 batch of data.
            else:
                inputs = batch["audio_features"].to(device)
                masks = batch["masks"].to(device)
                targets = batch["targets"].to(device).view(-1, 1) 

                optimizer.zero_grad()                    # To zero out the gradients.

                output,_ = model(inputs, masks)  # Predictions from 1 batch of data.

            loss = self.criterion(output, targets)         # Get the training loss.
            train_loss += loss.item()*inputs.size(0)

            loss.backward()                          # To backpropagate the error (gradients are computed).
            optimizer.step()                         # To update parameters based on current gradients.

        # train_loss = round(train_loss / len(data_loader.dataset), 4)
        # print("TRAIN"+" >> loss: %.4f " % train_loss)
        return train_loss


    def test(self, model, data_loader, mode):    

        model.eval()                                # Put the model in training mode.              
        num_correct = 0
        num_samples = 0
        val_loss = 0         

        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch in tqdm(data_loader):                    # Loop over all batches.
                if self.config.feature == 'spec':
                    inputs = batch["audio_features"].to(device)
                    targets = batch["targets"].to(device).view(-1, 1) 

                    output,_ = model(inputs)  # Predictions from 1 batch of data.
                else:
                    inputs = batch["audio_features"].to(device)
                    masks = batch["masks"].to(device)
                    targets = batch["targets"].to(device).view(-1, 1) 

                    output,_ = model(inputs, masks)  # Predictions from 1 batch of data.

                loss = self.criterion(output, targets)         # Get the training loss.
                val_loss += loss.item()*inputs.size(0)

                y_pred.append(output.cpu())
                y_true.append(targets.cpu())

        val_loss = round(val_loss / len(data_loader.dataset), 4)
        print(mode+" >> loss: %.4f " % val_loss)

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        results = self.metrics(pred, true)
        print('%s: >> ' %dict_to_str(results))

        results['Loss'] = round(val_loss, 4)

        return results


def run(config):
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True
    
    if config.dataset == 'mosi':
        key = 'Has0_acc_2'
    else:
        key = 'Mult_acc_2'
    
    if config.feature == 'spec':
        train_loader, test_loader, val_loader = AudioDataloader(batch_size=config.batch_size, dataset=config.dataset, feature=config.feature)
        if config.model_name == 'custom':
            model = AudioClassifier().to(device) 
    elif config.feature == 'smile':
        train_loader, test_loader, val_loader = AudioDataloader(batch_size=config.batch_size, dataset=config.dataset, feature=config.feature)
        model = AudioTransformers().to(device)
    else:
        train_loader, test_loader, val_loader = AudioDataloader(batch_size=config.batch_size, dataset=config.dataset, feature=config.feature)
        if config.dataset == 'mosi': 
            model = data2vec().to(device)
            for param in model.data2vec_model.feature_extractor.parameters():
                param.requires_grad = False
        else:
            model = hubert().to(device)
            for param in model.hubert_model.feature_extractor.parameters():
                param.requires_grad = False    
    trainer = audio_trainer(config)
    
    lowest_eval_loss = 100
    highest_eval_acc = 0
    epoch = 0
    best_epoch = 0
    while True:
        print('---------------------EPOCH: ', epoch, '--------------------')
        epoch += 1
        trainer.train(model=model, data_loader=train_loader)
        eval_results = trainer.test(model, val_loader,"VAL")
#         trainer.test(model, test_loader,"TEST")
        if eval_results['Loss']<lowest_eval_loss:
            lowest_eval_loss = eval_results['Loss']
            torch.save(model.state_dict(), config.model_save_path+'loss.pth')
            best_epoch = epoch
        if eval_results[key]>=highest_eval_acc:
            highest_eval_acc = eval_results[key]
            torch.save(model.state_dict(), config.model_save_path+'acc.pth')
        if epoch - best_epoch >= config.early_stop:
            break
    model.load_state_dict(torch.load(config.model_save_path+'acc.pth'))        
    test_results_loss = trainer.test(model, test_loader,"TEST")
    print('%s: >> ' %('TEST (highest val acc) ') + dict_to_str(test_results_loss))
    
    model.load_state_dict(torch.load(config.model_save_path+'loss.pth'))
    test_results_acc = trainer.test(model, test_loader,"TEST")
    print('%s: >> ' %('TEST (lowest val loss) ') + dict_to_str(test_results_acc))