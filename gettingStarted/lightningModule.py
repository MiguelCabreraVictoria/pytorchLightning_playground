"https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#lightning-hooks"

import os
import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L

"""
__init__ -> Define init method to initialize the model
forward() -> To run data through the model
training_step() 
validation_step()
test_step()
predict_step()
    - Inference refers to the process of using a machine learning model (task) to make predictions on new data
    - By default, runs the forward method
    - If you want to perform inference with the system, you can add a foward method
    - return self(batch) -> This will call the forward method

configure_optimizers() -> To define the optimizer and learning rate

.to(device) -> LightningModule will automatically move the model to the correct device

"""

class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder

    
    def training_step(self, batch, batch_idx):
        """
        Under the hood Lightning

        1. Clear the gradients : optimizer.zero_grad()
        2. Apply backpropagation : loss.backward()
        3. Update parameters : optimizer.step()
        """
        x, _ = batch
        x = x.view(x.size(0), -1) # Flatten the image
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x) 
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True) # Log epoch-level metrics
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


encoder = nn.Sequential(nn.Linear(28*28, 64),
                        nn.ReLU(),
                        nn.Linear(64, 3))

decoder = nn.Sequential(nn.Linear(3, 64),
                        nn.ReLU(),
                        nn.Linear(64, 28*28))

# autoencoder = LitAutoEncoder(encoder, decoder)

# # define the dataset

# dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
# train_loader = utils.data.DataLoader(dataset)

# # Train the model
# trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
# trainer.fit(autoencoder, train_loader)


# save the model
checkpoint = "./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
autoencoder = LitAutoEncoder.load_from_checkpoint(checkpoint, encoder=encoder, decoder=decoder)

encoder = autoencoder.encoder
encoder.eval()

fake_image_batch = torch.randn(4, 28*28, device=autoencoder.device)
embeddings = encoder(fake_image_batch)
print("⚡" * 20, "\nPredictions (4 image embeddings):\n", embeddings, "\n", "⚡" * 20)


"""

Save hyperparameters: improve readability and reproducibility

self.save_hyperparameters(ignore[hyperparameter_name])

model.load_from_checkpoint(checkpoint_path, hyperparameter_name=value)

"""

""" 
Callbacks allow you to add arbitrary self-contained programs to your training

Allows you to design programs that encapsulate a full set of functionality

Built-in callbacks:

- BatchSizeFinder : Finds the largest batch size supported by a given model before encountering an OOM error
- DeviceStatsMonitor : Monitor and logs devices stats during training, validation, and testing stage
- EarlyStopping : Stop training when a metric has stopped improving
- LearningRateFinder: Find the optimal learning rate for your model
- ModelCheckpoint : Save the model after every epoch
- ProgressBar : Adds a progress bar to the training loop

"""