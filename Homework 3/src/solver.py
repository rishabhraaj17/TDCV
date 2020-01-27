import torch
import torch.optim as optim

from utils.train_utils import save_checkpoint, load_checkpoint

# TODO: Remove it after train and evaluation is done


@DeprecationWarning
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
            anchor, puller, pusher = data
            anchor, puller, pusher = anchor.float().to(device), puller.float().to(device), pusher.float().to(device)

            X = torch.cat([anchor[0], puller[0], pusher[0]]).permute(dims=[0, 3, 1, 2])

            optimizer.zero_grad()
            y_pred = model(X)
            triplet_loss, pair_loss = self.loss_function(y_pred)
            loss = triplet_loss + pair_loss
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            if batch % 10 == 0 and self.writer_train is not None:
                self.writer_train.add_scalar('Train/Triplet_Loss', triplet_loss.item(), batch)
                self.writer_train.add_scalar('Train/Pair_Loss', pair_loss.item(), batch)

        train_loss /= len(train_loader.dataset)
        # train_accuracy /= len(train_loader.dataset)

        if self.writer_train is not None:
            self.writer_train.add_scalar('Training Loss', train_loss, current_epoch)

        print(f'Epoch: {current_epoch}, Training Loss: {train_loss}')

    def validation_step(self, model, val_loader, current_epoch, device):
        model.eval()

        val_loss = 0
        val_accuracy = 0

        with torch.no_grad():
            for batch, data in enumerate(val_loader):
                anchor, puller, pusher = data
                anchor, puller, pusher = anchor.float().to(device), puller.float().to(device), pusher.float().to(device)

                X = torch.cat([anchor[0], puller[0], pusher[0]]).permute(dims=[0, 3, 1, 2])

                y_pred = model(X)
                triplet_loss, pair_loss = self.loss_function(y_pred)
                loss = triplet_loss + pair_loss
                val_loss += loss.item()

                if batch % 10 == 0 and self.writer_val is not None:
                    self.writer_val.add_scalar('Validation/Triplet_Loss', triplet_loss.item(), batch)
                    self.writer_val.add_scalar('Validation/Pair_Loss', pair_loss.item(), batch)

        val_loss /= len(val_loader.dataset)
        # val_accuracy /= len(val_loader.dataset)

        if self.writer_val is not None:
            self.writer_val.add_scalar('Validation Loss', val_loss, current_epoch)

        print(f'Epoch : {current_epoch}, Validation Loss: {val_loss}')
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
