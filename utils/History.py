import torch

class History(object):
    def __init__(self, epochs):
        self.epochs = epochs
        self.start_epoch = 0
        self.end_epoch = 0
        self.train_loss = torch.Tensor(epochs)
        self.valid_loss = torch.Tensor(epochs)
        self.model_dict = None 
        self.optim_dict = None

    def load_context(self, last_weights):
        self.model_dict = last_weights['model_dict']
        self.optim_dict = last_weights['optim_dict']
        self.start_epoch = int(last_weights.get('start_epoch', 1))
        self.end_epoch = self.start_epoch
        self.epochs += self.start_epoch
        self.train_loss = torch.Tensor(self.epochs)
        self.valid_loss = torch.Tensor(self.epochs)
        self.train_loss[:self.start_epoch] = last_weights['train_loss'][:self.start_epoch]
        self.valid_loss[:self.start_epoch] = last_weights['valid_loss'][:self.start_epoch]

    def add_context(self, new_context):
        self.model_dict = new_context['model_dict']
        self.optim_dict = new_context['optim_dict']
        self.train_loss[self.end_epoch] = new_context['train_loss']
        self.valid_loss[self.end_epoch] = new_context['valid_loss']
        self.end_epoch += 1

    def save_context(self, save_file):
        package = {
            'start_epoch': self.end_epoch,
            'model_dict': self.model_dict,
            'optim_dict': self.optim_dict,
            'train_loss':  self.train_loss,
            'valid_loss':  self.valid_loss
        }
        torch.save(package, save_file)
        



