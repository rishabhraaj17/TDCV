import torch
import torch.optim as optim

from utils.train_utils import save_checkpoint, load_checkpoint


class Solver(object):
    def __init__(self, loss_function, optim_args, optimizer=optim.Adam, writer_train=None, writer_val=None):
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.optim_args = optim_args
        self.writer_val = writer_val
        self.writer_train = writer_train

    def train_step(self, model, optimizer, train_loader, current_epoch, device):
        model.train()

        train_loss = 0
        train_accuracy = 0

        for batch, data in enumerate(train_loader):
            X, y, rect = data
            X, y, rect = X.to(device), y.to(device), rect.to(device)

            optimizer.zero_grad()
            y_pred = model(X)
            loss = self.loss_function(y_pred, rect)  # Check if we need to swap it
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)

        if self.writer_train is not None:
            self.writer_train.add_scalar('Training Loss', train_loss, current_epoch)
            self.writer_train.add_scalar('Training Accuracy', train_accuracy, current_epoch)

        print(f'Epoch: {current_epoch}, Training Loss: {train_loss}, Training Accuracy: {train_accuracy * 100}%')

    def validation_step(self, model, val_loader, current_epoch, device):
        model.eval()

        val_loss = 0
        val_accuracy = 0

        with torch.no_grad():
            for batch, data in enumerate(val_loader):
                X, y, rect = data
                X, y, rect = X.to(device), y.to(device), rect.to(device)

                y_pred = model(X)
                loss = self.loss_function(y_pred, rect)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)

        if self.writer_val is not None:
            self.writer_val.add_scalar('Validation Loss', val_loss, current_epoch)
            self.writer_val.add_scalar('Validation Accuracy', val_accuracy, current_epoch)

        print(f'Epoch : {current_epoch}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy * 100}%')
        return val_loss

    def train(self, model, train_loader, val_loader, num_epochs, save_path, save_model=False, save_state_dict=True,
              log_checkpoint=False, checkpoint_dir=None, resume_training=False, resume_checkpoint_file=None):
        optimizer = self.optimizer(model.parameters(), **self.optim_args)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True

        model = model.to(device)

        resume_epoch = 0
        if resume_training:
            model, optimizer, resume_epoch = load_checkpoint(resume_checkpoint_file, model, optimizer)
            print(f'Resuming Training from epoch {resume_epoch - 1}/{num_epochs} epochs')

        best_val_loss = 1e8

        if not resume_epoch:
            print('START TRAINING')

        for epoch in range(resume_epoch, num_epochs):
            self.train_step(model=model, train_loader=train_loader, current_epoch=epoch, optimizer=optimizer, device=device)
            val_loss = self.validation_step(model=model, val_loader=val_loader, current_epoch=epoch, device=device)

            if log_checkpoint:
                checkpoint_dict = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }

                save_checkpoint(checkpoint_dict, checkpoint_dir=checkpoint_dir)

            if save_model:
                if val_loss < best_val_loss:
                    model.save(save_path, save_state_dict=save_state_dict)

        print('TRAINING FINISHED')
