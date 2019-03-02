
# coding: utf-8

# In[1]:


import os, sys
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision
import torch.nn.modules.normalization as norm
from torch.autograd import Variable
from process_data import ALOVDataset
import torch.optim as optim
import numpy as np


# In[2]:


LSTM_SIZE = 512


# In[3]:


class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta


    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x


# In[4]:


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# In[5]:


#alexnet = models.alexnet(pretrained=True)
class alexnet_conv_layers(nn.Module):
    def __init__(self):
        super(alexnet_conv_layers, self).__init__()
        self.base_features = torchvision.models.alexnet(pretrained = True).features
        self.skip1 = nn.Sequential(
            nn.Conv2d(64, out_channels=16, kernel_size=1, stride=1),
            nn.PReLU(),
            Flatten()
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(192, out_channels=32, kernel_size=1, stride=1),
            nn.PReLU(),
            Flatten()
        )
        self.skip5 = nn.Sequential(
            nn.Conv2d(256, out_channels=64, kernel_size=1, stride=1),
            nn.PReLU(),
            Flatten()
        )
        self.conv6 = nn.Sequential(
            nn.Linear(37104 * 2, 2048),
            nn.ReLU()
        )
        
        # Freeze those weights
        for p in self.base_features.parameters():
            p.requires_grad = False
        

    def forward(self, x, y):
        layer_extractor_x = []
        layer_extractor_y = []
        for idx, model in enumerate(self.base_features):
            x = model(x)
            y = model(y)
            if idx in {2, 5, 11}: # layer output of conv1, conv2 , conv5(before pooling layer)
                layer_extractor_x.append(x)
                layer_extractor_y.append(y)
                
        x_out_flat = x.view(1, -1) #(1, 256, 6, 6) --> (1, 9216)
        x_out_skip1 = self.skip1(layer_extractor_x[0]) #(1, 64, 27, 27) -> (11664)
        x_out_skip2 = self.skip2(layer_extractor_x[1]) #(1, 192, 13, 13) -> (5408)
        x_out_skip5 = self.skip5(layer_extractor_x[2]) #(1, 256, 13, 13) -> (10816)
        x_out = torch.cat((x_out_skip1, x_out_skip2, x_out_skip5, x_out_flat), dim=1)
        
        y_out_flat = y.view(1, -1) #(1, 256, 6, 6) --> (1, 9216)
        y_out_skip1 = self.skip1(layer_extractor_y[0]) #(1, 64, 27, 27) -> (11664)
        y_out_skip2 = self.skip2(layer_extractor_y[1]) #(1, 192, 13, 13) -> (5408)
        y_out_skip5 = self.skip5(layer_extractor_y[2]) #(1, 256, 13, 13) -> (10816)
        y_out = torch.cat((y_out_skip1, y_out_skip2, y_out_skip5, y_out_flat), dim=1)
        
        final_out = torch.cat((x_out, y_out), dim=1)
        conv_out = self.conv6(final_out) # (1, 2048)
        return conv_out


# In[6]:

class Re3Net(nn.Module):
    def __init__(self):
        super(Re3Net,self).__init__()
        self.conv_layers = alexnet_conv_layers()
        
        #2048 from conv_layers? maybe 1024?
        self.lstm1 =nn.LSTMCell(2048, LSTM_SIZE)
        self.lstm2 = nn.LSTMCell(2048 + LSTM_SIZE, LSTM_SIZE)

        self.fc_final = nn.Linear(LSTM_SIZE,4)
        
        self.h1 = Variable(torch.rand(1, LSTM_SIZE)).cuda()
        self.c1 = Variable(torch.rand(1, LSTM_SIZE)).cuda()
        
        self.h2 = Variable(torch.rand(1, LSTM_SIZE)).cuda()
        self.c2 = Variable(torch.rand(1, LSTM_SIZE)).cuda()

    def init_hidden(self):
        self.h1 = Variable(torch.rand(1, LSTM_SIZE)).cuda()
        self.c1 = Variable(torch.rand(1, LSTM_SIZE)).cuda()
        
        self.h2 = Variable(torch.rand(1, LSTM_SIZE)).cuda()
        self.c2 = Variable(torch.rand(1, LSTM_SIZE)).cuda()

    def detach_hidden(self):
        self.h1 = self.h1.detach()
        self.c1 = self.c1.detach()
        
        self.h2 = self.h2.detach()
        self.c2 = self.c2.detach()

    def forward(self, x, y):
        out = self.conv_layers(x, y)

        lstm1_out, self.h1 = self.lstm1(out, (self.h1, self.c1))

        lstm2_in = torch.cat((out, lstm1_out), dim=1)

        lstm2_out, self.h2 = self.lstm2(lstm2_in, (self.h2, self.c2))

        out = self.fc_final(lstm2_out)
        return out

class Re3Net_(nn.Module):
    def __init__(self):
        super(Re3Net,self).__init__()
        self.conv_layers = alexnet_conv_layers()
        
        #2048 from conv_layers? maybe 1024?
        self.lstm1 =nn.LSTMCell(2048, LSTM_SIZE)
        self.lstm2 = nn.LSTMCell(2048 + LSTM_SIZE, LSTM_SIZE)
        self.h0 = Variable(torch.rand(1 ,LSTM_SIZE)).cuda()
        self.c0 = Variable(torch.rand(1 ,LSTM_SIZE)).cuda()
        self.fc_final = nn.Linear(LSTM_SIZE,4)

        #self.h0=Variable(torch.rand(1,LSTM_SIZE)).cuda()
        #self.c0=Variable(torch.rand(1,LSTM_SIZE)).cuda()

    def init_hidden(self):
        self.h0 = Variable(torch.rand(1, LSTM_SIZE)).cuda()
        self.c0 = Variable(torch.rand(1, LSTM_SIZE)).cuda()

    def forward(self, x, prev_LSTM_state=False):
        out = self.conv_layers(x)
        
        #h0 = Variable(torch.rand(x.shape[0],LSTM_SIZE)).cuda()
        #c0 = Variable(torch.rand(x.shape[0],LSTM_SIZE)).cuda()
        
        lstm_out, self.h0 = self.lstm1(out, (self.h0, self.c0))

        lstm2_in = torch.cat((out, lstm_out), dim=1)

        lstm2_out, h1 = self.lstm2(lstm2_in, (self.h0, self.c0))

        out = self.fc_final(lstm2_out)
        return out


# In[7]:


class RunningAverage():
    """A simple class that maintains the running average of a quantity
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


# In[8]:


def evaluate(model, dataloader, criterion, epoch):

    model.eval()
    dataset = dataloader.dataset
    total_loss = 0

    for i in range(64):
        sample = dataset[i]
        sample['currimg'] = sample['currimg'][None,:,:,:]
        sample['previmg'] = sample['previmg'][None,:,:,:]
        x1, x2 = sample['previmg'], sample['currimg']
        y = sample['currbb']

        if use_gpu:
            x1 = Variable(x1.cuda())
            x2 = Variable(x2.cuda())
            y = Variable(y.cuda(), requires_grad=False)
        else:
            x1 = Variable(x1)
            x2 = Variable(x2)
            y = Variable(y, requires_grad=False)

        output = model(x1, x2)
        #print(output.size()) # [1,4]
        #print(y.size()) # [4]
        output = output.view(4)
        loss = criterion(output, y)
        total_loss += loss.item()
        if i % 10 == 0:
            print('[validation] epoch = %d, i = %d, loss = %f' % (epoch, i, loss.item()))

    seq_loss = total_loss/64
    return seq_loss


# In[10]:

# Adjust learning rate during training
def adjust_learning_rate(optimizer):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.1
        print("Change learning rate to: ", param_group['lr'])


def train_model(net, dataloader, optim, loss_function, num_epochs):

    dataset_size = dataloader.dataset.len
    for epoch in range(num_epochs):
        if epoch != 0 and epoch % 5 == 0:
            adjust_learning_rate(optim)

        net.train()
        curr_loss = 0.0

        # currently training on just ALOV dataset
        i = 0
        for data in dataloader:

            x1, x2, y = data['previmg'], data['currimg'], data['currbb']
            if use_gpu:
                x1, x2, y = Variable(x1.cuda()), Variable(x2.cuda()), Variable(y.cuda(), requires_grad=False)
            else:
                x1, x2, y = Variable(x1), Variable(x2), Variable(y, requires_grad=False)

            optim.zero_grad()

            output = net(x1.detach(), x2.detach())
            #print(output.size()) # [1,4]
            #print(y.size()) # [4]
            loss = loss_function(output, y)

            loss.backward(retain_graph=True)
            optim.step()
            if i%20 == 0:
                print('[training] epoch = %d, i = %d/%d, loss = %f' % (epoch, i, dataset_size, loss.item()) )
                sys.stdout.flush()
            if i%32 == 0:
                net.init_hidden()
            i = i + 1
            curr_loss += loss.item()
        epoch_loss = curr_loss / dataset_size
        print('Loss: {:.4f}'.format(epoch_loss))
        
        path = save_directory + '_batch_' + str(epoch) + '_loss_' + str(round(epoch_loss, 3)) + '.pth'
        torch.save(net.state_dict(), path)

        val_loss = evaluate(net, dataloader, loss_function, epoch)
        print('Validation Loss: {:.4f}'.format(val_loss))
    return net


# In[11]:


# Convert numpy arrays to torch tensors
class ToTensor(object):
    def __call__(self, sample):
        prev_img, curr_img = sample['previmg'], sample['currimg']
        # swap color axis because numpy image: H x W x C ; torch image: C X H X W
        prev_img = prev_img.transpose((2, 0, 1))
        curr_img = curr_img.transpose((2, 0, 1))
        if 'currbb' in sample:
            currbb = sample['currbb']
            return {'previmg': torch.from_numpy(prev_img).float(),
                    'currimg': torch.from_numpy(curr_img).float(),
                    'currbb': torch.from_numpy(currbb).float()
                    }
        else:
            return {'previmg': torch.from_numpy(prev_img).float(),
                    'currimg': torch.from_numpy(curr_img).float()
                    }


# To normalize the data points
class Normalize(object):
    def __call__(self, sample):

        prev_img, curr_img = sample['previmg'], sample['currimg']
        self.mean = [104, 117, 123]
        prev_img = prev_img.astype(float)
        curr_img = curr_img.astype(float)
        prev_img -= np.array(self.mean).astype(float)
        curr_img -= np.array(self.mean).astype(float)

        if 'currbb' in sample:
            currbb = sample['currbb']
            currbb = currbb*(10./227);
            return {'previmg': prev_img,
                    'currimg': curr_img,
                    'currbb': currbb}
        else:
            return {'previmg': prev_img,
                    'currimg': curr_img
}
transform = transforms.Compose([Normalize(), ToTensor()])


# In[12]:


save_directory = 'pytorch_model/'
save_model_step = 5
learning_rate = 0.001
use_gpu = True
num_epochs = 100


# In[13]:

alov = ALOVDataset('/media/arg_ws3/5E703E3A703E18EB/data/alov/imagedata++/', '/media/arg_ws3/5E703E3A703E18EB/data/alov/alov300++_rectangleAnnotation_full/', transform)
dataloader = DataLoader(alov, batch_size = 1)


# In[14]:


net = Re3Net().cuda()
loss_function = torch.nn.L1Loss(size_average=False).cuda()
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.0005)
#net.load_state_dict(torch.load("/home/arg_ws3/re3_tracking/saved_models_pytorch/_batch_1_loss_2.952.pth"))

# In[15]:


if os.path.exists(save_directory):
    print('Directory %s already exists', save_directory)
else:
    os.makedirs(save_directory)
    print('Create directory: %s', save_directory)


# In[16]:


net = train_model(net, dataloader, optimizer, loss_function, num_epochs)


# In[31]:


torch.cuda.empty_cache()


# In[82]:


a = torch.empty(1, 4, dtype=torch.float)

b = torch.empty(4, dtype=torch.float)

print(a.size(), b.size())

c = a.view(4)
print(c.size())

