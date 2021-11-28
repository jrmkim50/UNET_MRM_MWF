from Hang.utils_u_groupnorm_pytorchLightning import unetConv3d, unetUp3d
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
import Hang.unet_new_utils as unet_new_utils

class unet3d(pl.LightningModule):
    def __init__(self,
                 learning_rate,
                 decay_factor = 0.2,
                 feature_scale = 2, 
                 n_classes = 6, 
                 is_deconv = False, 
                 in_channels = 6, 
                 is_groupnorm = True,
    ):
        super(unet3d, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_groupnorm = is_groupnorm
        self.feature_scale = feature_scale
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv3d(self.in_channels, filters[0], self.is_groupnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size = (2,2,2), padding = (1,1,1))
        self.conv2 = unetConv3d(filters[0], filters[1], self.is_groupnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size = (2,2,2), padding = (1,1,1))
        self.conv3 = unetConv3d(filters[1], filters[2], self.is_groupnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size = (2,2,2), padding = (1,1,1))
        self.conv4 = unetConv3d(filters[2], filters[3], self.is_groupnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size = (2,2,2), padding = (1,1,1))
        
        self.center = unetConv3d(filters[3], filters[4], self.is_groupnorm)
        
        # upsampling
        self.up_concat4 = unetUp3d(filters[4]+filters[3], filters[3], self.is_deconv)
        self.up_concat3 = unetUp3d(filters[3]+filters[2], filters[2], self.is_deconv)
        self.up_concat2 = unetUp3d(filters[2]+filters[1], filters[1], self.is_deconv)
        self.up_concat1 = unetUp3d(filters[1]+filters[0], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1) 
        
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        up4 = self.up_concat4(conv4, self.center(maxpool4))
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        criterion1 = nn.L1Loss()
        loss = criterion1(y_hat, y)
        tensorboard_logs = {'loss': loss }
        return {'loss': loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        criterion1 = nn.L1Loss()
        loss = criterion1(y_hat, y)
        tensorboard_logs = {'val_loss': loss.cpu()}
        return {'val_loss': loss.cpu(), 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss.cpu()}
        return {'avg_val_loss': avg_loss.cpu(), 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.decay_factor, patience=6),
            'monitor': 'avg_val_loss', 
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]
    
    
class unet3d_warmup(pl.LightningModule):
    def __init__(self,
                 learning_rate,
                 decay_factor = 0.2,
                 feature_scale = 2, 
                 n_classes = 6, 
                 is_deconv = False, 
                 in_channels = 6, 
                 is_groupnorm = True,
    ):
        super(unet3d_warmup, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_groupnorm = is_groupnorm
        self.feature_scale = feature_scale
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv3d(self.in_channels, filters[0], self.is_groupnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size = (2,2,2), padding = (1,1,1))
        self.conv2 = unetConv3d(filters[0], filters[1], self.is_groupnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size = (2,2,2), padding = (1,1,1))
        self.conv3 = unetConv3d(filters[1], filters[2], self.is_groupnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size = (2,2,2), padding = (1,1,1))
        self.conv4 = unetConv3d(filters[2], filters[3], self.is_groupnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size = (2,2,2), padding = (1,1,1))
        
        self.center = unetConv3d(filters[3], filters[4], self.is_groupnorm)
        
        # upsampling
        self.up_concat4 = unetUp3d(filters[4]+filters[3], filters[3], self.is_deconv)
        self.up_concat3 = unetUp3d(filters[3]+filters[2], filters[2], self.is_deconv)
        self.up_concat2 = unetUp3d(filters[2]+filters[1], filters[1], self.is_deconv)
        self.up_concat1 = unetUp3d(filters[1]+filters[0], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1) 
        
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        up4 = self.up_concat4(conv4, self.center(maxpool4))
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        criterion1 = nn.L1Loss()
        loss = criterion1(y_hat, y)
        tensorboard_logs = {'loss': loss }
        return {'loss': loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        criterion1 = nn.L1Loss()
        loss = criterion1(y_hat, y)
        tensorboard_logs = {'val_loss': loss.cpu()}
        return {'val_loss': loss.cpu(), 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss.cpu()}
        return {'avg_val_loss': avg_loss.cpu(), 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        
        def lr(epoch):
            # 0: 0.1^5, 1: 0.1^4, 2: 0.1^3, 3: 0.1^2, 4: 0.1^1, 5: 0.1^0, 
            # 6: 0.95^1, ...
            if epoch <= 5:
                lr_scale = 0.1**(5-epoch)
            else:
                lr_scale = 0.95**(epoch-5)
            return lr_scale
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr)
        
        return [optimizer], [scheduler]
    
class unet3d_tuning(pl.LightningModule):
    def __init__(self,
                 learning_rate,
                 feature_scale = 2, 
                 n_classes = 6,
                 in_channels = 6,
                 kernel_size = (3,3,1), 
                 padding = (1,1,0),
                 is_batchnorm = True
    ):
        super(unet3d_tuning, self).__init__()
        self.learning_rate = learning_rate

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / feature_scale) for x in filters]

        # downsampling
        self.conv1 = unet_new_utils.unetConv3d(in_channels, filters[0], 
                                               is_batchnorm, kernel_size, padding)
        self.maxpool1 = nn.MaxPool3d(2)
        self.conv2 = unet_new_utils.unetConv3d(filters[0], filters[1], 
                                               is_batchnorm, kernel_size, padding)
        self.maxpool2 = nn.MaxPool3d(2)
        self.conv3 = unet_new_utils.unetConv3d(filters[1], filters[2], 
                                               is_batchnorm, kernel_size, padding)
        self.maxpool3 = nn.MaxPool3d(2)
        self.conv4 = unet_new_utils.unetConv3d(filters[2], filters[3], 
                                               is_batchnorm, kernel_size, padding)
        self.maxpool4 = nn.MaxPool3d(2)
        
        self.center = unet_new_utils.unetConv3d(filters[3], filters[4], 
                                                is_batchnorm, kernel_size, padding)
        
        # upsampling
        self.up_concat4 = unet_new_utils.unetUp3d(filters[4]+filters[3], filters[3],
                                                  is_batchnorm, kernel_size, padding)
        self.up_concat3 = unet_new_utils.unetUp3d(filters[3]+filters[2], filters[2], 
                                                  is_batchnorm, kernel_size, padding)
        self.up_concat2 = unet_new_utils.unetUp3d(filters[2]+filters[1], filters[1], 
                                                  is_batchnorm, kernel_size, padding)
        self.up_concat1 = unet_new_utils.unetUp3d(filters[1]+filters[0], filters[0], 
                                                  is_batchnorm, kernel_size, padding)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1) 
        
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        up4 = self.up_concat4(conv4, self.center(maxpool4))
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        criterion1 = nn.L1Loss()
        loss = criterion1(y_hat, y)
        tensorboard_logs = {'loss': loss }
        return {'loss': loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        criterion1 = nn.L1Loss()
        loss = criterion1(y_hat, y)
        tensorboard_logs = {'val_loss': loss.cpu()}
        return {'val_loss': loss.cpu(), 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss.cpu()}
        return {'avg_val_loss': avg_loss.cpu(), 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        
        def lr(epoch):
            # 0: 0.1^5, 1: 0.1^4, 2: 0.1^3, 3: 0.1^2, 4: 0.1^1, 5: 0.1^0, 
            # 6: 0.95^1, ...
            if epoch <= 5:
                lr_scale = 0.1**(5-epoch)
            else:
                lr_scale = 0.95**(epoch-5)
            return lr_scale
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr)
        
        return [optimizer], [scheduler]