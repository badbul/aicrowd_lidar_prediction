import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import yaml

class Tnet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, input):
        # input.shape == (bs,n,3)
        bs = input.size(0)
        xb = F.relu(self.bn1(self.conv1(input)))
        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = F.relu(self.bn3(self.conv3(xb)))
        pool = nn.MaxPool1d(xb.size(-1))(xb)
        flat = nn.Flatten(1)(pool)
        xb = F.relu(self.bn4(self.fc1(flat)))
        xb = F.relu(self.bn5(self.fc2(xb)))

        # initialize as identity
        init = torch.eye(self.k, requires_grad=True).repeat(bs, 1, 1)
        if xb.is_cuda:
            init = init.cuda()
        matrix = self.fc3(xb).view(-1, self.k, self.k) + init
        return matrix


class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(3, 64, 1)

        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, input):
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(input, 1, 2), matrix3x3).transpose(1, 2)

        xb = F.relu(self.bn1(self.conv1(xb)))

        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb, 1, 2), matrix64x64).transpose(1, 2)

        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = self.bn3(self.conv3(xb))
        xb = nn.MaxPool1d(xb.size(-1))(xb)
        output = nn.Flatten(1)(xb)
        return output, matrix3x3, matrix64x64

class PointNet(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.transform = Transform()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        xb, matrix3x3, matrix64x64 = self.transform(input)
        xb = F.relu(self.bn1(self.fc1(xb)))
        xb = F.relu(self.bn2(self.dropout(self.fc2(xb))))
        output = self.fc3(xb)
        #return self.logsoftmax(output), matrix3x3, matrix64x64
        return output, matrix3x3, matrix64x64

def pointnetloss(outputs, labels, m3x3, m64x64, alpha = 0.0001):
    #criterion = torch.nn.NLLLoss()
    criterion = torch.nn.L1Loss()
    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs,1,1)
    if outputs.is_cuda:
        id3x3=id3x3.cuda()
        id64x64=id64x64.cuda()
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
    return criterion(outputs, labels) #+ alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)


class PointNetModel(LightningModule):

    def __init__(self, config=None):
        super().__init__()
        self.save_hyperparameters()
        self.config = config  # prepare_config(self, config)
        self.net = PointNet(classes=1)

    def forward(self, inputs):
        return self.net(inputs)


    def training_step(self, batch, batch_idx):
        inputs, labels = batch['pointcloud'], batch['label']
        inputs = inputs.transpose(1, 2)
        outputs, m3x3, m64x64 = self.net(inputs)
        loss = pointnetloss(outputs, labels, m3x3, m64x64)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch['pointcloud'], batch['label']
        inputs = inputs.transpose(1, 2)
        outputs, m3x3, m64x64 = self.net(inputs)
        loss = pointnetloss(outputs, labels, m3x3, m64x64)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_epoch_end(self, outputs):
        # self.coco_evaluator.accumulate()
        # self.coco_evaluator.summarize()
        # coco main metric
        # metric = self.coco_evaluator.coco_eval["bbox"].stats[0]
        # metric = torch.as_tensor(metric)
        # tensorboard_logs = {"main_score": metric}
        # return {
        #     "val_loss": metric,
        #     "log": tensorboard_logs,
        #     "progress_bar": tensorboard_logs,
        # }
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config['lr'])
        #'''
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.8,
                patience=2,
                threshold=0.0001,
                threshold_mode='abs',
                cooldown=0,
                min_lr=1.e-8,
                eps=1.e-8,
            ),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        #'''
        return [optimizer], [scheduler]

