import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from .dropout import DropMask,createMask
from . import cnn
import csv
import numpy as np
import random
import importlib


class LSTMcell_untied(torch.nn.Module):
    def __init__(self,
                 *,
                 inputSize,
                 hiddenSize,
                 train=True,
                 dr=0.5,
                 drMethod='gal+sem',
                 gpu=0):
        super(LSTMcell_untied,self).__init__()
        self.inputSize=inputSize
        self.hiddenSize=inputSize
        self.dr=dr

        self.w_xi=Parameter(torch.Tensor(hiddenSize,inputSize))
        self.w_xf=Parameter(torch.Tensor(hiddenSize,inputSize))
        self.w_xo=Parameter(torch.Tensor(hiddenSize,inputSize))
        self.w_xc=Parameter(torch.Tensor(hiddenSize,inputSize))

        self.w_hi=Parameter(torch.Tensor(hiddenSize,hiddenSize))
        self.w_hf=Parameter(torch.Tensor(hiddenSize,hiddenSize))
        self.w_ho=Parameter(torch.Tensor(hiddenSize,hiddenSize))
        self.w_hc=Parameter(torch.Tensor(hiddenSize,hiddenSize))

        self.b_i=Parameter(torch.Tensor(hiddenSize))
        self.b_f=Parameter(torch.Tensor(hiddenSize))
        self.b_o=Parameter(torch.Tensor(hiddenSize))
        self.b_c=Parameter(torch.Tensor(hiddenSize))

        self.drMethod=drMethod.split('+')
        self.gpu=gpu
        self.train=train
        if gpu>=0:
            self=self.cuda(gpu)
            self.is_cuda=True
        else:
            self.is_cuda=False
        self.reset_parameters()

    def reset_parameters(self):
        std=1.0/math.sqrt(self.hiddenSize)
        for w in self.parameters():
            w.data.uniform_(-std,std)

    def init_mask(self,x,h,c):
        self.maskX_i=createMask(x,self.dr)
        self.maskX_f=createMask(x,self.dr)
        self.maskX_c=createMask(x,self.dr)
        self.maskX_o=createMask(x,self.dr)

        self.maskH_i=createMask(h,self.dr)
        self.maskH_f=createMask(h,self.dr)
        self.maskH_c=createMask(h,self.dr)
        self.maskH_o=createMask(h,self.dr)

        self.maskC=createMask(c,self.dr)

        self.maskW_xi=createMask(self.w_xi,self.dr)
        self.maskW_xf=createMask(self.w_xf,self.dr)
        self.maskW_xc=createMask(self.w_xc,self.dr)
        self.maskW_xo=createMask(self.w_xo,self.dr)
        self.maskW_hi=createMask(self.w_hi,self.dr)
        self.maskW_hf=createMask(self.w_hf,self.dr)
        self.maskW_hc=createMask(self.w_hc,self.dr)
        self.maskW_ho=createMask(self.w_ho,self.dr)

    def forward(self,x,hidden):
        h0,c0=hidden
        doDrop=self.training and self.dr>0.0

        if doDrop:
            self.init_mask(x,h0,c0)

        if doDrop and 'drH' in self.drMethod:
            h0_i=h0.mul(self.maskH_i)
            h0_f=h0.mul(self.maskH_f)
            h0_c=h0.mul(self.maskH_c)
            h0_o=h0.mul(self.maskH_o)
        else:
            h0_i=h0
            h0_f=h0
            h0_c=h0
            h0_o=h0

        if doDrop and 'drX' in self.drMethod:
            x_i=x.mul(self.maskX_i)
            x_f=x.mul(self.maskX_f)
            x_c=x.mul(self.maskX_c)
            x_o=x.mul(self.maskX_o)
        else:
            x_i=x
            x_f=x
            x_c=x
            x_o=x

        if doDrop and 'drW' in self.drMethod:
            w_xi=self.w_xi.mul(self.maskW_xi)
            w_xf=self.w_xf.mul(self.maskW_xf)
            w_xc=self.w_xc.mul(self.maskW_xc)
            w_xo=self.w_xo.mul(self.maskW_xo)
            w_hi=self.w_hi.mul(self.maskW_hi)
            w_hf=self.w_hf.mul(self.maskW_hf)
            w_hc=self.w_hc.mul(self.maskW_hc)
            w_ho=self.w_ho.mul(self.maskW_ho)
        else:
            w_xi=self.w_xi
            w_xf=self.w_xf
            w_xc=self.w_xc
            w_xo=self.w_xo
            w_hi=self.w_hi
            w_hf=self.w_hf
            w_hc=self.w_hc
            w_ho=self.w_ho

        gate_i=F.linear(x_i,w_xi)+F.linear(h0_i,w_hi)+self.b_i
        gate_f=F.linear(x_f,w_xf)+F.linear(h0_f,w_hf)+self.b_f
        gate_c=F.linear(x_c,w_xc)+F.linear(h0_c,w_hc)+self.b_c
        gate_o=F.linear(x_o,w_xo)+F.linear(h0_o,w_ho)+self.b_o

        gate_i=F.sigmoid(gate_i)
        gate_f=F.sigmoid(gate_f)
        gate_c=F.tanh(gate_c)
        gate_o=F.sigmoid(gate_o)

        if doDrop and 'drC' in self.drMethod:
            gate_c=gate_c.mul(self.maskC)

        c1=(gate_f*c0)+(gate_i*gate_c)
        h1=gate_o*F.tanh(c1)

        return h1,c1


class LSTMcell_tied(torch.nn.Module):
    def __init__(self,
                 *,
                 inputSize,
                 hiddenSize,
                 mode='train',
                 dr=0.5,
                 drMethod='drX+drW+drC',
                 gpu=1):
        super(LSTMcell_tied,self).__init__()

        self.inputSize=inputSize
        self.hiddenSize=hiddenSize
        self.dr=dr

        self.w_ih=Parameter(torch.Tensor(hiddenSize*4,inputSize))
        self.w_hh=Parameter(torch.Tensor(hiddenSize*4,hiddenSize))
        self.b_ih=Parameter(torch.Tensor(hiddenSize*4))
        self.b_hh=Parameter(torch.Tensor(hiddenSize*4))

        self.drMethod=drMethod.split('+')
        self.gpu=gpu
        self.mode=mode
        if mode=='train':
            self.train(mode=True)
        elif mode=='test':
            self.train(mode=False)
        elif mode=='drMC':
            self.train(mode=False)

        if gpu>=0:
            self=self.cuda()
            self.is_cuda=True
        else:
            self.is_cuda=False
        self.reset_parameters()

    def reset_parameters(self):
        stdv=1.0/math.sqrt(self.hiddenSize)
        for weight in self.parameters():
            weight.data.uniform_(-stdv,stdv)

    def reset_mask(self,x,h,c):
        self.maskX=createMask(x,self.dr)
        self.maskH=createMask(h,self.dr)
        self.maskC=createMask(c,self.dr)
        self.maskW_ih=createMask(self.w_ih,self.dr)
        self.maskW_hh=createMask(self.w_hh,self.dr)

    def forward(self,x,hidden,*,resetMask=True,doDropMC=False):
        if self.dr>0 and (doDropMC is True or self.training is True):
            doDrop=True
        else:
            doDrop=False

        batchSize=x.size(0)
        h0,c0=hidden
        if h0 is None:
            h0=x.new_zeros(batchSize,self.hiddenSize,requires_grad=False)
        if c0 is None:
            c0=x.new_zeros(batchSize,self.hiddenSize,requires_grad=False)

        if self.dr>0 and self.training is True and resetMask is True:
            self.reset_mask(x,h0,c0)

        if doDrop is True and 'drH' in self.drMethod:
            h0=DropMask.apply(h0,self.maskH,True)

        if doDrop is True and 'drX' in self.drMethod:
            x=DropMask.apply(x,self.maskX,True)

        if doDrop is True and 'drW' in self.drMethod:
            w_ih=DropMask.apply(self.w_ih,self.maskW_ih,True)
            w_hh=DropMask.apply(self.w_hh,self.maskW_hh,True)
        else:
            # self.w are parameters, while w are not
            w_ih=self.w_ih
            w_hh=self.w_hh

        gates=F.linear(x,w_ih,self.b_ih)+\
              F.linear(h0,w_hh,self.b_hh)
        gate_i,gate_f,gate_c,gate_o=gates.chunk(4,1)

        gate_i=torch.sigmoid(gate_i)
        gate_f=torch.sigmoid(gate_f)
        gate_c=torch.tanh(gate_c)
        gate_o=torch.sigmoid(gate_o)

        if self.training is True and 'drC' in self.drMethod:
            gate_c=gate_c.mul(self.maskC)

        c1=(gate_f*c0)+(gate_i*gate_c)
        h1=gate_o*torch.tanh(c1)

        return h1,c1


class CudnnLstm(torch.nn.Module):
    def __init__(self,*,inputSize,hiddenSize,dr=0.5,drMethod='drW',
                 gpu=0):
        super(CudnnLstm,self).__init__()
        self.inputSize=inputSize
        self.hiddenSize=hiddenSize
        self.dr=dr
        self.w_ih=Parameter(torch.Tensor(hiddenSize*4,inputSize))
        self.w_hh=Parameter(torch.Tensor(hiddenSize*4,hiddenSize))
        self.b_ih=Parameter(torch.Tensor(hiddenSize*4))
        self.b_hh=Parameter(torch.Tensor(hiddenSize*4))
        self._all_weights=[['w_ih','w_hh','b_ih','b_hh']]
        # Release portability adaptation: permit CPU-side architecture
        # inspection. Scientific execution still uses the original CUDNN path.
        if torch.cuda.is_available():
            self.cuda()

        self.reset_mask()
        self.reset_parameters()

    def _apply(self,fn):
        ret=super(CudnnLstm,self)._apply(fn)
        return ret

    def __setstate__(self,d):
        super(CudnnLstm,self).__setstate__(d)
        self.__dict__.setdefault('_data_ptrs',[])
        if 'all_weights' in d:
            self._all_weights=d['all_weights']
        if isinstance(self._all_weights[0][0],str):
            return
        self._all_weights=[['w_ih','w_hh','b_ih','b_hh']]

    def reset_mask(self):
        self.maskW_ih=createMask(self.w_ih,self.dr)
        self.maskW_hh=createMask(self.w_hh,self.dr)

    def reset_parameters(self):
        stdv=1.0/math.sqrt(self.hiddenSize)
        for weight in self.parameters():
            weight.data.uniform_(-stdv,stdv)

    def forward(self,input,hx=None,cx=None,doDropMC=False,dropoutFalse=False):
        # dropoutFalse: it will ensure doDrop is false, unless doDropMC is true
        if dropoutFalse and (not doDropMC):
            doDrop=False
        elif self.dr>0 and (doDropMC is True or self.training is True):
            doDrop=True
        else:
            doDrop=False

        batchSize=input.size(1)

        if hx is None:
            hx=input.new_zeros(
                1,batchSize,self.hiddenSize,requires_grad=False)
        if cx is None:
            cx=input.new_zeros(
                1,batchSize,self.hiddenSize,requires_grad=False)

        # cuDNN backend - disabled flat weight
        # handle = torch.backends.cudnn.get_handle()
        if doDrop is True:
            self.reset_mask()
            weight=[
                DropMask.apply(self.w_ih,self.maskW_ih,True),
                DropMask.apply(self.w_hh,self.maskW_hh,True),self.b_ih,
                self.b_hh
            ]
        else:
            weight=[self.w_ih,self.w_hh,self.b_ih,self.b_hh]

        # output, hy, cy, reserve, new_weight_buf = torch._cudnn_rnn(
        #     input, weight, 4, None, hx, cx, torch.backends.cudnn.CUDNN_LSTM,
        #     self.hiddenSize, 1, False, 0, self.training, False, (), None)
        if torch.__version__<"1.8":
            output,hy,cy,reserve,new_weight_buf=torch._cudnn_rnn(
                input,weight,4,None,hx,cx,2,  # 2 means LSTM
                self.hiddenSize,1,False,0,self.training,False,(),None)
        else:
            output,hy,cy,reserve,new_weight_buf=torch._cudnn_rnn(
                input,weight,4,None,hx,cx,2,  # 2 means LSTM
                self.hiddenSize,0,1,False,0,self.training,False,(),None)
        return output,(hy,cy)

    @property
    def all_weights(self):
        return [[getattr(self,weight) for weight in weights]
                for weights in self._all_weights]


class CNN1dkernel(torch.nn.Module):
    def __init__(self,
                 *,
                 ninchannel=1,
                 nkernel=3,
                 kernelSize=3,
                 stride=1,
                 padding=0):
        super(CNN1dkernel,self).__init__()
        self.cnn1d=torch.nn.Conv1d(
            in_channels=ninchannel,
            out_channels=nkernel,
            kernel_size=kernelSize,
            padding=padding,
            stride=stride,
        )

    def forward(self,x):
        output=F.relu(self.cnn1d(x))
        return output


class BasicBlock(nn.Module):
    expansion=1

    def __init__(self,in_channels,out_channels,stride=1,downsample=None):
        super(BasicBlock,self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.downsample=downsample
        self.stride=stride

    def forward(self,x):
        identity=x

        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)

        if self.downsample is not None:
            identity=self.downsample(x)

        out+=identity
        out=self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,block,layers,num_classes=10):
        super(ResNet,self).__init__()
        self.in_channels=4
        self.conv1=nn.Conv2d(1,4,kernel_size=5,stride=4,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(4)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(kernel_size=5,stride=4,padding=1)
        self.layer1=self._make_layer(block,4,layers[0])
        self.layer2=self._make_layer(block,64,layers[1],stride=2)
        self.layer3=self._make_layer(block,128,layers[2],stride=2)
        self.layer4=self._make_layer(block,512,layers[3],stride=2)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512*block.expansion,num_classes)

    def _make_layer(self,block,out_channels,blocks,stride=1):
        downsample=None
        if stride!=1 or self.in_channels!=out_channels*block.expansion:
            downsample=nn.Sequential(
                nn.Conv2d(self.in_channels,out_channels*block.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels*block.expansion),
            )

        layers=[]
        layers.append(block(self.in_channels,out_channels,stride,downsample))
        self.in_channels=out_channels*block.expansion
        for _ in range(1,blocks):
            layers.append(block(self.in_channels,out_channels))

        return nn.Sequential(*layers)

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)

        x=self.avgpool(x)
        x=torch.flatten(x,1)
        x=self.fc(x)

        return x


def ResNet18():
    return ResNet(BasicBlock,[2,2,2,2])


def ResNet34():
    return ResNet(BasicBlock,[3,4,6,3])


class BasicBlockv2(nn.Module):
    expansion=1

    def __init__(self,in_channels,out_channels,stride=1,downsample=None):
        super(BasicBlockv2,self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.downsample=downsample
        self.stride=stride

    def forward(self,x):
        identity=x

        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)

        if self.downsample is not None:
            identity=self.downsample(x)

        out+=identity
        out=self.relu(out)

        return out


class ResNetv2(nn.Module):
    def __init__(self,block,layers,num_classes=10):
        super(ResNetv2,self).__init__()
        self.in_channels=4
        self.conv1=nn.Conv2d(1,4,kernel_size=3,stride=2,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(4)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1=self._make_layer(block,4,layers[0])
        self.layer2=self._make_layer(block,64,layers[1],stride=2)
        self.layer3=self._make_layer(block,128,layers[2],stride=2)
        self.layer4=self._make_layer(block,512,layers[3],stride=2)
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512*block.expansion,num_classes)

    def _make_layer(self,block,out_channels,blocks,stride=1):
        downsample=None
        if stride!=1 or self.in_channels!=out_channels*block.expansion:
            downsample=nn.Sequential(
                nn.Conv2d(self.in_channels,out_channels*block.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels*block.expansion),
            )

        layers=[]
        layers.append(block(self.in_channels,out_channels,stride,downsample))
        self.in_channels=out_channels*block.expansion
        for _ in range(1,blocks):
            layers.append(block(self.in_channels,out_channels))

        return nn.Sequential(*layers)

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)

        x=self.avgpool(x)
        x=torch.flatten(x,1)
        x=self.fc(x)

        return x


def ResNet18v2():
    return ResNetv2(BasicBlockv2,[2,2,2,2])


class CudnnLstmModel(torch.nn.Module):
    def __init__(self,*,nx,ny,hiddenSize,dr=0.5):
        super(CudnnLstmModel,self).__init__()
        self.nx=nx
        self.ny=ny
        self.hiddenSize=hiddenSize
        self.ct=0
        self.nLayer=1
        self.linearIn=torch.nn.Linear(nx,hiddenSize)
        self.lstm=CudnnLstm(
            inputSize=hiddenSize,hiddenSize=hiddenSize,dr=dr)
        self.linearOut=torch.nn.Linear(hiddenSize,ny)
        self.gpu=1
        # self.drtest = torch.nn.Dropout(p=0.4)

    def forward(self,x,doDropMC=False,dropoutFalse=False):
        x0=F.relu(self.linearIn(x))
        outLSTM,(hn,cn)=self.lstm(x0,doDropMC=doDropMC,dropoutFalse=dropoutFalse)
        # outLSTMdr = self.drtest(outLSTM)
        out=self.linearOut(outLSTM)
        return out


# class CudnnLstmModel_weights(torch.nn.Module):
#     def __init__(self,*,nx,ny,hiddenSize,dr=0.5):
#         super(CudnnLstmModel_weights,self).__init__()
#         self.nx=nx-6
#         self.ny=1
#         self.hiddenSize=hiddenSize
#         self.ct=0
#         self.nLayer=1
#         self.linearIn=torch.nn.Linear(self.nx,hiddenSize)
#         self.lstm=CudnnLstm(
#             inputSize=hiddenSize,hiddenSize=hiddenSize,dr=dr)
#         self.linearOut=torch.nn.Linear(hiddenSize,6)
#         self.gpu=1
#         # self.drtest = torch.nn.Dropout(p=0.4)
#
#     def forward(self,x,doDropMC=False,dropoutFalse=False):
#         x0=F.relu(self.linearIn(x[:,:,6:]))
#         outLSTM,(hn,cn)=self.lstm(x0,doDropMC=doDropMC,dropoutFalse=dropoutFalse)
#         # outLSTMdr = self.drtest(outLSTM)
#         out_ori=self.linearOut(outLSTM)
#
#         out = (out_ori[:,:,0]*out_ori[:,:,0]\
#                 +out_ori[:,:,1]*out_ori[:,:,1]\
#                 +out_ori[:,:,2]*out_ori[:,:,2]\
#                 +out_ori[:,:,3]*out_ori[:,:,3]\
#                 +out_ori[:,:,4]*out_ori[:,:,4]\
#                 +out_ori[:,:,5]*out_ori[:,:,5])/6
#         return out[:,:,None]


class CNN1dLSTMmodel(torch.nn.Module):
    def __init__(self,*,nx,ny,nobs,hiddenSize,
                 nkernel=(10,5),kernelSize=(3,3),stride=(2,1),dr=0.5,poolOpt=None):
        # two convolutional layer
        super(CNN1dLSTMmodel,self).__init__()
        self.nx=nx
        self.ny=ny
        self.obs=nobs
        self.hiddenSize=hiddenSize
        nlayer=len(nkernel)
        self.features=nn.Sequential()
        ninchan=1
        Lout=nobs
        for ii in range(nlayer):
            ConvLayer=CNN1dkernel(
                ninchannel=ninchan,nkernel=nkernel[ii],kernelSize=kernelSize[ii],stride=stride[ii])
            self.features.add_module('CnnLayer%d'%(ii+1),ConvLayer)
            ninchan=nkernel[ii]
            Lout=cnn.calConvSize(lin=Lout,kernel=kernelSize[ii],stride=stride[ii])
            self.features.add_module('Relu%d'%(ii+1),nn.ReLU())
            if poolOpt is not None:
                self.features.add_module('Pooling%d'%(ii+1),nn.MaxPool1d(poolOpt[ii]))
                Lout=cnn.calPoolSize(lin=Lout,kernel=poolOpt[ii])
        self.Ncnnout=int(Lout*nkernel[-1])  # total CNN feature number after convolution
        Nf=self.Ncnnout+nx
        self.linearIn=torch.nn.Linear(Nf,hiddenSize)
        self.lstm=CudnnLstm(
            inputSize=hiddenSize,hiddenSize=hiddenSize,dr=dr)
        self.linearOut=torch.nn.Linear(hiddenSize,ny)
        self.gpu=1

    def forward(self,x,z,doDropMC=False):
        nt,ngrid,nobs=z.shape
        z=z.view(nt*ngrid,1,nobs)
        z0=self.features(z)
        # z0 = (ntime*ngrid) * nkernel * sizeafterconv
        z0=z0.view(nt,ngrid,self.Ncnnout)
        x0=torch.cat((x,z0),dim=2)
        x0=F.relu(self.linearIn(x0))
        outLSTM,(hn,cn)=self.lstm(x0,doDropMC=doDropMC)
        out=self.linearOut(outLSTM)
        # out = rho/time * batchsize * Ntargetvar
        return out


class CNN1dLSTMInmodel(torch.nn.Module):
    # Directly add the CNN extracted features into LSTM inputSize
    def __init__(self,*,nx,ny,nobs,hiddenSize,
                 nkernel=(10,5),kernelSize=(3,3),stride=(2,1),dr=0.5,poolOpt=None,cnndr=0.0):
        # two convolutional layer
        super(CNN1dLSTMInmodel,self).__init__()
        self.nx=nx
        self.ny=ny
        self.obs=nobs
        self.hiddenSize=hiddenSize
        nlayer=len(nkernel)
        self.features=nn.Sequential()
        ninchan=1
        Lout=nobs
        for ii in range(nlayer):
            ConvLayer=CNN1dkernel(
                ninchannel=ninchan,nkernel=nkernel[ii],kernelSize=kernelSize[ii],stride=stride[ii])
            self.features.add_module('CnnLayer%d'%(ii+1),ConvLayer)
            if cnndr!=0.0:
                self.features.add_module('dropout%d'%(ii+1),nn.Dropout(p=cnndr))
            ninchan=nkernel[ii]
            Lout=cnn.calConvSize(lin=Lout,kernel=kernelSize[ii],stride=stride[ii])
            self.features.add_module('Relu%d'%(ii+1),nn.ReLU())
            if poolOpt is not None:
                self.features.add_module('Pooling%d'%(ii+1),nn.MaxPool1d(poolOpt[ii]))
                Lout=cnn.calPoolSize(lin=Lout,kernel=poolOpt[ii])
        self.Ncnnout=int(Lout*nkernel[-1])  # total CNN feature number after convolution
        Nf=self.Ncnnout+hiddenSize
        self.linearIn=torch.nn.Linear(nx,hiddenSize)
        self.lstm=CudnnLstm(
            inputSize=Nf,hiddenSize=hiddenSize,dr=dr)
        self.linearOut=torch.nn.Linear(hiddenSize,ny)
        self.gpu=1

    def forward(self,x,z,doDropMC=False):
        nt,ngrid,nobs=z.shape
        z=z.view(nt*ngrid,1,nobs)
        z0=self.features(z)
        # z0 = (ntime*ngrid) * nkernel * sizeafterconv
        z0=z0.view(nt,ngrid,self.Ncnnout)
        x=F.relu(self.linearIn(x))
        x0=torch.cat((x,z0),dim=2)
        outLSTM,(hn,cn)=self.lstm(x0,doDropMC=doDropMC)
        out=self.linearOut(outLSTM)
        # out = rho/time * batchsize * Ntargetvar
        return out


class MultiInv_HBVTDModel_HAND_CNN1dLSTMInmodel_pjli(torch.nn.Module):
    # Directly add the CNN extracted features into LSTM inputSize
    def __init__(self,*,nx,ny,nobs,hiddenSize,
                 nkernel=(10,5),kernelSize=(3,3),stride=(2,1),dr=0.5,poolOpt=None,cnndr=0.0):
        # two convolutional layer
        super(MultiInv_HBVTDModel_HAND_CNN1dLSTMInmodel_pjli,self).__init__()
        self.nx=nx
        self.ny=ny
        self.obs=nobs
        self.hiddenSize=hiddenSize
        nlayer=len(nkernel)
        self.features=nn.Sequential()
        ninchan=2
        Lout=nobs
        for ii in range(nlayer):
            ConvLayer=CNN1dkernel(
                ninchannel=ninchan,nkernel=nkernel[ii],kernelSize=kernelSize[ii],stride=stride[ii])
            self.features.add_module('CnnLayer%d'%(ii+1),ConvLayer)
            if cnndr!=0.0:
                self.features.add_module('dropout%d'%(ii+1),nn.Dropout(p=cnndr))
            ninchan=nkernel[ii]
            Lout=cnn.calConvSize(lin=Lout,kernel=kernelSize[ii],stride=stride[ii])
            self.features.add_module('Relu%d'%(ii+1),nn.ReLU())
            if poolOpt is not None:
                self.features.add_module('Pooling%d'%(ii+1),nn.MaxPool1d(poolOpt[ii]))
                Lout=cnn.calPoolSize(lin=Lout,kernel=poolOpt[ii])
        self.Ncnnout=int(Lout*nkernel[-1])  # total CNN feature number after convolution
        Nf=self.Ncnnout+self.nx
        self.linearIn=torch.nn.Linear(nx,hiddenSize)
        self.lstm=CudnnLstmModel(
            nx=Nf,ny=ny,hiddenSize=hiddenSize,dr=dr)

    def forward(self,x,z,zd,doDropMC=False):

        ntstep=z.shape[0]
        ngage=z.shape[1]

        # zd = ngrid*nVar*nchan add a channel dimension
        z0=self.features(zd)
        # z0 = (ngrid) * nkernel * sizeafterconv
        z0_time=z0.view(ngage,self.Ncnnout).repeat(ntstep,1,1)

        z_add=torch.cat((z,z0_time),dim=2)

        out=self.lstm(z_add)
        return out

# class CudnnLstmModel_weights(torch.nn.Module):
#     # Directly add the CNN extracted features into LSTM inputSize
#     def __init__(self,*,nx,ny,hiddenSize,dr=0.8):
#         # two convolutional layer
#         super(CudnnLstmModel_weights,self).__init__()
#         self.nx=nx
#         self.ny=ny
#         self.dr=dr
#         self.hiddenSize=hiddenSize
#
#         Nf=self.nx
#         self.ANN=AnnModel_static(
#             nx=Nf,ny=ny,hiddenSize=hiddenSize,dr=self.dr)
#
#         # # Add the Dropout layer
#         # self.dropout = nn.Dropout(p=dr)
#
#     def forward(self,x,doDropMC=False,dropoutFalse=False):
#         weights_ori=self.ANN(x[0,:,self.ny:])
#         weights = torch.sigmoid(weights_ori)*100
#
#         zd=x[:,:,:self.ny]
#         weights_tile1 = weights[None].repeat(zd.shape[0],1,1)
#         # weights_tile2 = torch.sum(weights,dim=1)[None,:][:,:,None].repeat(zd.shape[0],1,zd.shape[1])
#         # weights_tile2=torch.sum(weights,dim=1).unsqueeze(0).unsqueeze(-1).repeat(zd.shape[0],1,zd.shape[1])
#         weights_tile2_ori=torch.sum(weights,dim=1).unsqueeze(0).repeat(zd.shape[0],1)
#
#         weights_tile2 = torch.clamp(weights_tile2_ori,min=0.01)
#
#
#         out = torch.sum(weights_tile1*zd,dim=-1)/weights_tile2
#         return out.unsqueeze(-1)

class CudnnLstmModel_weights(torch.nn.Module):
    # Directly add the CNN extracted features into LSTM inputSize
    def __init__(self,*,nx,ny,hiddenSize,dr=0.5):
        # two convolutional layer
        super(CudnnLstmModel_weights,self).__init__()
        self.nx=nx
        self.ny=ny
        self.dr=dr
        self.hiddenSize=hiddenSize

        Nf=self.nx
        self.ANN=AnnModel_static(
            nx=Nf,ny=ny,hiddenSize=hiddenSize,dr=self.dr)

        # # Add the Dropout layer
        # self.dropout = nn.Dropout(p=dr)

    def forward(self,x,doDropMC=False,dropoutFalse=False):
        weights_ori=self.ANN(x[0,:,self.ny:])
        weights = torch.sigmoid(weights_ori)*100

        # zd=x[:,:,:6]
        # weights_tile1 = weights[None].repeat(zd.shape[0],1,1)
        # # weights_tile2 = torch.sum(weights,dim=1)[None,:][:,:,None].repeat(zd.shape[0],1,zd.shape[1])
        # # weights_tile2=torch.sum(weights,dim=1).unsqueeze(0).unsqueeze(-1).repeat(zd.shape[0],1,zd.shape[1])
        # weights_tile2_ori=torch.sum(weights,dim=1).unsqueeze(0).repeat(zd.shape[0],1)
        #
        # weights_tile2 = torch.clamp(weights_tile2_ori,min=0.01)
        #
        #
        # out = torch.sum(weights_tile1*zd,dim=-1)/weights_tile2
        return weights[None]


class MultiInv_HBVTDModel_HAND_CNN1dLSTMInmodel_pjli_shap(torch.nn.Module):
    # Directly add the CNN extracted features into LSTM inputSize
    def __init__(self,*,nx,ny,nobs,hiddenSize,
                 nkernel=(10,5),kernelSize=(3,3),stride=(2,1),dr=0.5,poolOpt=None,cnndr=0.0):
        # two convolutional layer
        super(MultiInv_HBVTDModel_HAND_CNN1dLSTMInmodel_pjli_shap,self).__init__()
        self.nx=nx
        self.ny=ny
        self.obs=nobs
        self.hiddenSize=hiddenSize
        nlayer=len(nkernel)
        self.features=nn.Sequential()
        ninchan=2
        Lout=nobs
        for ii in range(nlayer):
            ConvLayer=CNN1dkernel(
                ninchannel=ninchan,nkernel=nkernel[ii],kernelSize=kernelSize[ii],stride=stride[ii])
            self.features.add_module('CnnLayer%d'%(ii+1),ConvLayer)
            if cnndr!=0.0:
                self.features.add_module('dropout%d'%(ii+1),nn.Dropout(p=cnndr))
            ninchan=nkernel[ii]
            Lout=cnn.calConvSize(lin=Lout,kernel=kernelSize[ii],stride=stride[ii])
            self.features.add_module('Relu%d'%(ii+1),nn.ReLU())
            if poolOpt is not None:
                self.features.add_module('Pooling%d'%(ii+1),nn.MaxPool1d(poolOpt[ii]))
                Lout=cnn.calPoolSize(lin=Lout,kernel=poolOpt[ii])
        self.Ncnnout=int(Lout*nkernel[-1])  # total CNN feature number after convolution
        Nf=self.Ncnnout+self.nx
        self.linearIn=torch.nn.Linear(nx,hiddenSize)
        self.lstm=CudnnLstmModel(
            nx=Nf,ny=ny,hiddenSize=hiddenSize,dr=dr)

    def forward(self,z_int,doDropMC=False):
        (x,z,zd) = z_int
        ntstep=z.shape[0]
        ngage=z.shape[1]

        # zd = ngrid*nVar*nchan add a channel dimension
        z0=self.features(zd)
        # z0 = (ngrid) * nkernel * sizeafterconv
        z0_time=z0.view(ngage,self.Ncnnout).repeat(ntstep,1,1)

        z_add=torch.cat((z,z0_time),dim=2)

        out=self.lstm(z_add)
        return out


class CNN1dLCmodel(torch.nn.Module):
    # add the CNN extracted features into original LSTM input, then pass through linear layer
    def __init__(self,*,nx,ny,nobs,hiddenSize,
                 nkernel=(10,5),kernelSize=(3,3),stride=(2,1),dr=0.5,poolOpt=None,cnndr=0.0):
        # two convolutional layer
        super(CNN1dLCmodel,self).__init__()
        self.nx=nx
        self.ny=ny
        self.obs=nobs
        self.hiddenSize=hiddenSize
        nlayer=len(nkernel)
        self.features=nn.Sequential()
        ninchan=1  # need to modify the hardcode: 4 for smap and 1 for FDC
        Lout=nobs
        for ii in range(nlayer):
            ConvLayer=CNN1dkernel(
                ninchannel=ninchan,nkernel=nkernel[ii],kernelSize=kernelSize[ii],stride=stride[ii])
            self.features.add_module('CnnLayer%d'%(ii+1),ConvLayer)
            if cnndr!=0.0:
                self.features.add_module('dropout%d'%(ii+1),nn.Dropout(p=cnndr))
            ninchan=nkernel[ii]
            Lout=cnn.calConvSize(lin=Lout,kernel=kernelSize[ii],stride=stride[ii])
            self.features.add_module('Relu%d'%(ii+1),nn.ReLU())
            if poolOpt is not None:
                self.features.add_module('Pooling%d'%(ii+1),nn.MaxPool1d(poolOpt[ii]))
                Lout=cnn.calPoolSize(lin=Lout,kernel=poolOpt[ii])
        self.Ncnnout=int(Lout*nkernel[-1])  # total CNN feature number after convolution
        Nf=self.Ncnnout+nx
        self.linearIn=torch.nn.Linear(Nf,hiddenSize)
        self.lstm=CudnnLstm(
            inputSize=hiddenSize,hiddenSize=hiddenSize,dr=dr)
        self.linearOut=torch.nn.Linear(hiddenSize,ny)
        self.gpu=1

    def forward(self,x,z,doDropMC=False):
        # z = ngrid*nVar add a channel dimension
        ngrid=z.shape[0]
        rho,BS,Nvar=x.shape
        if len(z.shape)==2:  # for FDC, else 3 dimension for smap
            z=torch.unsqueeze(z,dim=1)
        z0=self.features(z)
        # z0 = (ngrid) * nkernel * sizeafterconv
        z0=z0.view(ngrid,self.Ncnnout).repeat(rho,1,1)
        x=torch.cat((x,z0),dim=2)
        x0=F.relu(self.linearIn(x))
        outLSTM,(hn,cn)=self.lstm(x0,doDropMC=doDropMC)
        out=self.linearOut(outLSTM)
        # out = rho/time * batchsize * Ntargetvar
        return out


class CNN1dLCInmodel(torch.nn.Module):
    # Directly add the CNN extracted features into LSTM inputSize
    def __init__(self,*,nx,ny,nobs,hiddenSize,
                 nkernel=(10,5),kernelSize=(3,3),stride=(2,1),dr=0.5,poolOpt=None,cnndr=0.0):
        # two convolutional layer
        super(CNN1dLCInmodel,self).__init__()
        self.nx=nx
        self.ny=ny
        self.obs=nobs
        self.hiddenSize=hiddenSize
        nlayer=len(nkernel)
        self.features=nn.Sequential()
        ninchan=1
        Lout=nobs
        for ii in range(nlayer):
            ConvLayer=CNN1dkernel(
                ninchannel=ninchan,nkernel=nkernel[ii],kernelSize=kernelSize[ii],stride=stride[ii])
            self.features.add_module('CnnLayer%d'%(ii+1),ConvLayer)
            if cnndr!=0.0:
                self.features.add_module('dropout%d'%(ii+1),nn.Dropout(p=cnndr))
            ninchan=nkernel[ii]
            Lout=cnn.calConvSize(lin=Lout,kernel=kernelSize[ii],stride=stride[ii])
            self.features.add_module('Relu%d'%(ii+1),nn.ReLU())
            if poolOpt is not None:
                self.features.add_module('Pooling%d'%(ii+1),nn.MaxPool1d(poolOpt[ii]))
                Lout=cnn.calPoolSize(lin=Lout,kernel=poolOpt[ii])
        self.Ncnnout=int(Lout*nkernel[-1])  # total CNN feature number after convolution
        Nf=self.Ncnnout+hiddenSize
        self.linearIn=torch.nn.Linear(nx,hiddenSize)
        self.lstm=CudnnLstm(
            inputSize=Nf,hiddenSize=hiddenSize,dr=dr)
        self.linearOut=torch.nn.Linear(hiddenSize,ny)
        self.gpu=1

    def forward(self,x,z,doDropMC=False):
        # z = ngrid*nVar add a channel dimension
        ngrid,nobs=z.shape
        rho,BS,Nvar=x.shape
        z=torch.unsqueeze(z,dim=1)
        z0=self.features(z)
        # z0 = (ngrid) * nkernel * sizeafterconv
        z0=z0.view(ngrid,self.Ncnnout).repeat(rho,1,1)
        x=F.relu(self.linearIn(x))
        x0=torch.cat((x,z0),dim=2)
        outLSTM,(hn,cn)=self.lstm(x0,doDropMC=doDropMC)
        out=self.linearOut(outLSTM)
        # out = rho/time * batchsize * Ntargetvar
        return out


class CudnnInvLstmModel(torch.nn.Module):
    # using cudnnLstm to extract features from SMAP observations
    def __init__(self,*,nx,ny,hiddenSize,ninv,nfea,hiddeninv,dr=0.5,drinv=0.5):
        # two LSTM
        super(CudnnInvLstmModel,self).__init__()
        self.nx=nx
        self.ny=ny
        self.hiddenSize=hiddenSize
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.lstminv=CudnnLstmModel(
            nx=ninv,ny=nfea,hiddenSize=hiddeninv,dr=drinv)
        self.lstm=CudnnLstmModel(
            nx=nfea+nx,ny=ny,hiddenSize=hiddenSize,dr=dr)
        self.gpu=1

    def forward(self,x,z,doDropMC=False):
        Gen=self.lstminv(z)
        dim=x.shape;
        nt=dim[0]
        invpara=Gen[-1,:,:].repeat(nt,1,1)
        x1=torch.cat((x,invpara),dim=2)
        out=self.lstm(x1)
        # out = rho/time * batchsize * Ntargetvar
        return out


class LstmCloseModel(torch.nn.Module):
    def __init__(self,*,nx,ny,hiddenSize,dr=0.5,fillObs=True):
        super(LstmCloseModel,self).__init__()
        self.nx=nx
        self.ny=ny
        self.hiddenSize=hiddenSize
        self.ct=0
        self.nLayer=1
        # self.linearIn=torch.nn.Linear(nx+1,hiddenSize)
        self.linearIn=torch.nn.Linear(nx+ny,hiddenSize)
        # self.lstm = CudnnLstm(
        #     inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.lstm=LSTMcell_tied(
            inputSize=hiddenSize,hiddenSize=hiddenSize,dr=dr,drMethod='drW')
        self.linearOut=torch.nn.Linear(hiddenSize,ny)
        self.gpu=1
        self.fillObs=fillObs

    def forward(self,x,y=None):
        nt,ngrid,nx=x.shape
        # yt=torch.zeros(ngrid,1).cuda()
        yt=torch.zeros(ngrid,self.ny).cuda()
        out=torch.zeros(nt,ngrid,self.ny).cuda()
        ht=None
        ct=None
        resetMask=True
        for t in range(nt):
            if self.fillObs is True:
                ytObs=y[t,:,:]
                mask=ytObs==ytObs
                yt[mask]=ytObs[mask]
            xt=torch.cat((x[t,:,:],yt),1)
            x0=F.relu(self.linearIn(xt))
            ht,ct=self.lstm(x0,hidden=(ht,ct),resetMask=resetMask)
            yt=self.linearOut(ht)
            resetMask=False
            out[t,:,:]=yt
        return out


class AnnModel(torch.nn.Module):
    def __init__(self,*,nx,ny,hiddenSize):
        super(AnnModel,self).__init__()
        self.hiddenSize=hiddenSize
        self.i2h=nn.Linear(nx,hiddenSize)
        self.h2h=nn.Linear(hiddenSize,hiddenSize)
        self.h2o=nn.Linear(hiddenSize,ny)
        self.ny=ny

    def forward(self,x,y=None):
        nt,ngrid,nx=x.shape
        yt=torch.zeros(ngrid,1).cuda()
        out=torch.zeros(nt,ngrid,self.ny).cuda()
        for t in range(nt):
            xt=x[t,:,:]
            ht=F.relu(self.i2h(xt))
            ht2=self.h2h(ht)
            yt=self.h2o(ht2)
            out[t,:,:]=yt
        return out

class AnnModel_static(torch.nn.Module):
    def __init__(self,*,nx,ny,hiddenSize,dr=0.5):
        super(AnnModel_static,self).__init__()
        self.hiddenSize=hiddenSize
        self.i2h=nn.Linear(nx,hiddenSize)
        self.h2h=nn.Linear(hiddenSize,hiddenSize)
        self.h2o=nn.Linear(hiddenSize,ny)
        self.ny=ny
        self.dr=dr
        self.doDrop=self.training and self.dr>0.0
        # Add the Dropout layer
        self.dropout = nn.Dropout(p=self.dr)

    def forward(self,x,y=None):
        ngrid,nx=x.shape
        out=torch.zeros(ngrid,self.ny).cuda()
        ht=F.relu(self.i2h(x))
        if self.doDrop:
            ht2=self.dropout(F.relu(self.h2h(ht)))
        else:
            ht2=F.relu(self.h2h(ht))
        out=self.h2o(ht2)
        return out


class AnnCloseModel(torch.nn.Module):
    def __init__(self,*,nx,ny,hiddenSize,fillObs=True):
        super(AnnCloseModel,self).__init__()
        self.hiddenSize=hiddenSize
        self.i2h=nn.Linear(nx+1,hiddenSize)
        self.h2h=nn.Linear(hiddenSize,hiddenSize)
        self.h2o=nn.Linear(hiddenSize,ny)
        self.fillObs=fillObs
        self.ny=ny

    def forward(self,x,y=None):
        nt,ngrid,nx=x.shape
        yt=torch.zeros(ngrid,1).cuda()
        out=torch.zeros(nt,ngrid,self.ny).cuda()
        for t in range(nt):
            if self.fillObs is True:
                ytObs=y[t,:,:]
                mask=ytObs==ytObs
                yt[mask]=ytObs[mask]
            xt=torch.cat((x[t,:,:],yt),1)
            ht=F.relu(self.i2h(xt))
            ht2=self.h2h(ht)
            yt=self.h2o(ht2)
            out[t,:,:]=yt
        return out


class LstmCnnCond(nn.Module):
    def __init__(self,
                 *,
                 nx,
                 ny,
                 ct,
                 opt=1,
                 hiddenSize=64,
                 cnnSize=32,
                 cp1=(64,3,2),
                 cp2=(128,5,2),
                 dr=0.5):
        super(LstmCnnCond,self).__init__()

        # opt == 1: cnn output as initial state of LSTM (h0)
        # opt == 2: cnn output as additional output of LSTM
        # opt == 3: cnn output as constant input of LSTM

        if opt==1:
            cnnSize=hiddenSize

        self.nx=nx
        self.ny=ny
        self.ct=ct
        self.ctRm=False
        self.hiddenSize=hiddenSize
        self.opt=opt

        self.cnn=cnn.Cnn1d(nx=nx,nt=ct,cnnSize=cnnSize,cp1=cp1,cp2=cp2)

        self.lstm=CudnnLstm(
            inputSize=hiddenSize,hiddenSize=hiddenSize,dr=dr)
        if opt==3:
            self.linearIn=torch.nn.Linear(nx+cnnSize,hiddenSize)
        else:
            self.linearIn=torch.nn.Linear(nx,hiddenSize)
        if opt==2:
            self.linearOut=torch.nn.Linear(hiddenSize+cnnSize,ny)
        else:
            self.linearOut=torch.nn.Linear(hiddenSize,ny)

    def forward(self,x,xc):
        # x- [nt,ngrid,nx]
        x1=xc
        x1=self.cnn(x1)
        x2=x
        if self.opt==1:
            x2=F.relu(self.linearIn(x2))
            x2,(hn,cn)=self.lstm(x2,hx=x1[None,:,:])
            x2=self.linearOut(x2)
        elif self.opt==2:
            x1=x1[None,:,:].repeat(x2.shape[0],1,1)
            x2=F.relu(self.linearIn(x2))
            x2,(hn,cn)=self.lstm(x2)
            x2=self.linearOut(torch.cat([x2,x1],2))
        elif self.opt==3:
            x1=x1[None,:,:].repeat(x2.shape[0],1,1)
            x2=torch.cat([x2,x1],2)
            x2=F.relu(self.linearIn(x2))
            x2,(hn,cn)=self.lstm(x2)
            x2=self.linearOut(x2)

        return x2


class LstmCnnForcast(nn.Module):
    def __init__(self,
                 *,
                 nx,
                 ny,
                 ct,
                 opt=1,
                 hiddenSize=64,
                 cnnSize=32,
                 cp1=(64,3,2),
                 cp2=(128,5,2),
                 dr=0.5):
        super(LstmCnnForcast,self).__init__()

        if opt==1:
            cnnSize=hiddenSize

        self.nx=nx
        self.ny=ny
        self.ct=ct
        self.ctRm=True
        self.hiddenSize=hiddenSize
        self.opt=opt
        self.cnnSize=cnnSize

        if opt==1:
            self.cnn=cnn.Cnn1d(
                nx=nx+1,nt=ct,cnnSize=cnnSize,cp1=cp1,cp2=cp2)
        if opt==2:
            self.cnn=cnn.Cnn1d(
                nx=1,nt=ct,cnnSize=cnnSize,cp1=cp1,cp2=cp2)

        self.lstm=CudnnLstm(
            inputSize=hiddenSize,hiddenSize=hiddenSize,dr=dr)
        self.linearIn=torch.nn.Linear(nx+cnnSize,hiddenSize)
        self.linearOut=torch.nn.Linear(hiddenSize,ny)

    def forward(self,x,y):
        # x- [nt,ngrid,nx]
        nt,ngrid,nx=x.shape
        ct=self.ct
        pt=nt-ct

        if self.opt==1:
            x1=torch.cat((y,x),dim=2)
        elif self.opt==2:
            x1=y

        x1out=torch.zeros([pt,ngrid,self.cnnSize]).cuda()
        for k in range(pt):
            x1out[k,:,:]=self.cnn(x1[k:k+ct,:,:])

        x2=x[ct:nt,:,:]
        x2=torch.cat([x2,x1out],2)
        x2=F.relu(self.linearIn(x2))
        x2,(hn,cn)=self.lstm(x2)
        x2=self.linearOut(x2)

        return x2


class CudnnLstmModel_R2P(torch.nn.Module):
    pass


class CpuLstmModel(torch.nn.Module):
    def __init__(self,*,nx,ny,hiddenSize,dr=0.5):
        super(CpuLstmModel,self).__init__()
        self.nx=nx
        self.ny=ny
        self.hiddenSize=hiddenSize
        self.ct=0
        self.nLayer=1
        self.linearIn=torch.nn.Linear(nx,hiddenSize)
        self.lstm=LSTMcell_tied(
            inputSize=hiddenSize,hiddenSize=hiddenSize,dr=dr,drMethod='drW',gpu=-1)
        self.linearOut=torch.nn.Linear(hiddenSize,ny)
        self.gpu=-1

    def forward(self,x,doDropMC=False):
        # x0 = F.relu(self.linearIn(x))
        # outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC)
        # out = self.linearOut(outLSTM)
        # return out
        nt,ngrid,nx=x.shape
        yt=torch.zeros(ngrid,1)
        out=torch.zeros(nt,ngrid,self.ny)
        ht=None
        ct=None
        resetMask=True
        for t in range(nt):
            xt=x[t,:,:]
            x0=F.relu(self.linearIn(xt))
            ht,ct=self.lstm(x0,hidden=(ht,ct),resetMask=resetMask)
            yt=self.linearOut(ht)
            resetMask=False
            out[t,:,:]=yt
        return out


def UH_conv(x,UH,viewmode=1):
    # UH is a vector indicating the unit hydrograph
    # the convolved dimension will be the last dimension
    # UH convolution is
    # Q(t)=\integral(x(\tao)*UH(t-\tao))d\tao
    # conv1d does \integral(w(\tao)*x(t+\tao))d\tao
    # hence we flip the UH
    # https://programmer.group/pytorch-learning-conv1d-conv2d-and-conv3d.html
    # view
    # x: [batch, var, time]
    # UH:[batch, var, uhLen]
    # batch needs to be accommodated by channels and we make use of groups
    # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    # https://pytorch.org/docs/stable/nn.functional.html

    mm=x.shape;
    nb=mm[0]
    m=UH.shape[-1]
    padd=m-1
    if viewmode==1:
        xx=x.view([1,nb,mm[-1]])
        w=UH.view([nb,1,m])
        groups=nb

    y=F.conv1d(xx,torch.flip(w,[2]),groups=groups,padding=padd,stride=1,bias=None)
    y=y[:,:,0:-padd]
    return y.view(mm)


def UH_gamma(a,b,lenF=10):
    # UH. a [time (same all time steps), batch, var]
    m=a.shape
    w=torch.zeros([lenF,m[1],m[2]])
    aa=F.relu(a[0:lenF,:,:]).view([lenF,m[1],m[2]])+0.1  # minimum 0.1. First dimension of a is repeat
    theta=F.relu(b[0:lenF,:,:]).view([lenF,m[1],m[2]])+0.5  # minimum 0.5
    t=torch.arange(0.5,lenF*1.0).view([lenF,1,1]).repeat([1,m[1],m[2]])
    t=t.cuda(aa.device)
    denom=(aa.lgamma().exp())*(theta**aa)
    mid=t**(aa-1)
    right=torch.exp(-t/theta)
    w=1/denom*mid*right
    w=w/w.sum(0)  # scale to 1 for each UH

    return w


class SimpAnn(torch.nn.Module):
    def __init__(self,*,nx,ny,hiddenSize):
        super(SimpAnn,self).__init__()
        self.hiddenSize=hiddenSize
        self.i2h=nn.Linear(nx,hiddenSize)
        self.h2h=nn.Linear(hiddenSize,hiddenSize)
        self.h2o=nn.Linear(hiddenSize,ny)
        self.ny=ny

    def forward(self,x):
        ht=F.relu(self.i2h(x))
        ht2=F.relu(self.h2h(ht))
        out=F.relu(self.h2o(ht2))
        return out


class HBVMul(torch.nn.Module):
    """Multi-component HBV model implemented in PyTorch by Dapeng Feng"""

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMul,self).__init__()

    def forward(self,x,parameters,mu,muwts,rtwts,bufftime=0,outstate=False,routOpt=False,comprout=False,
                corrwts=None,pcorr=None):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     X: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2]
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     corrwts:P correction weights; pcorr: P correction opt
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET

        PRECS=1e-5  # keep the numerical calculation stable

        # Initialization for warm up states
        if bufftime>0:
            with torch.no_grad():
                xinit=x[0:bufftime,:,:]
                initmodel=HBVMul()
                Qsinit,SNOWPACK,MELTWATER,SM,SUZ,SLZ=initmodel(xinit,parameters,mu,muwts,rtwts,
                                                               bufftime=0,outstate=True,routOpt=False,comprout=False,
                                                               corrwts=corrwts,pcorr=pcorr)
        else:
            # Without warm-up bufftime=0, initialize state variables with zeros
            Ngrid=x.shape[1]
            SNOWPACK=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).cuda()
            MELTWATER=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).cuda()
            SM=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).cuda()
            SUZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).cuda()
            SLZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).cuda()
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()

        P=x[bufftime:,:,0]
        Nstep,Ngrid=P.size()
        if pcorr is not None:
            parPCORR=pcorr[0]+corrwts[:,0]*(pcorr[1]-pcorr[0])
            P=parPCORR.unsqueeze(0).repeat(Nstep,1)*P
            # print('P corrected')

        Pm=P.unsqueeze(2).repeat(1,1,mu)  # precip
        T=x[bufftime:,:,1]
        Tm=T.unsqueeze(2).repeat(1,1,mu)  # temperature
        ETpot=x[bufftime:,:,2]
        ETpm=ETpot.unsqueeze(2).repeat(1,1,mu)  # potential ET

        ## scale the parameters to real vales
        parascaLst=[[1,6],[50,1000],[0.05,0.9],[0.01,0.5],[0.001,0.2],[0.2,1],
                    [0,10],[0,100],[-2.5,2.5],[0.5,10],[0,0.1],[0,0.2]]  # HBV para
        routscaLst=[[0,2.9],[0,6.5]]  # routing para
        # dim of each para is [Nbasin*Ncomponent]
        parBETA=parascaLst[0][0]+parameters[:,0,:]*(parascaLst[0][1]-parascaLst[0][0])
        parFC=parascaLst[1][0]+parameters[:,1,:]*(parascaLst[1][1]-parascaLst[1][0])
        parK0=parascaLst[2][0]+parameters[:,2,:]*(parascaLst[2][1]-parascaLst[2][0])
        parK1=parascaLst[3][0]+parameters[:,3,:]*(parascaLst[3][1]-parascaLst[3][0])
        parK2=parascaLst[4][0]+parameters[:,4,:]*(parascaLst[4][1]-parascaLst[4][0])
        parLP=parascaLst[5][0]+parameters[:,5,:]*(parascaLst[5][1]-parascaLst[5][0])
        parPERC=parascaLst[6][0]+parameters[:,6,:]*(parascaLst[6][1]-parascaLst[6][0])
        parUZL=parascaLst[7][0]+parameters[:,7,:]*(parascaLst[7][1]-parascaLst[7][0])
        parTT=parascaLst[8][0]+parameters[:,8,:]*(parascaLst[8][1]-parascaLst[8][0])
        parCFMAX=parascaLst[9][0]+parameters[:,9,:]*(parascaLst[9][1]-parascaLst[9][0])
        parCFR=parascaLst[10][0]+parameters[:,10,:]*(parascaLst[10][1]-parascaLst[10][0])
        parCWH=parascaLst[11][0]+parameters[:,11,:]*(parascaLst[11][1]-parascaLst[11][0])

        # Initialize time series of model variables
        Qsimmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).cuda()
        ETmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).cuda()

        # Output the three simulated components of total Q
        Qsimmu0=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).cuda()
        Qsimmu1=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).cuda()
        Qsimmu2=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).cuda()

        for t in range(Nstep):
            # Separate precipitation into liquid and solid components
            PRECIP=Pm[t,:,:]  # need to check later, seems repeating with line 52
            RAIN=torch.mul(PRECIP,(Tm[t,:,:]>=parTT).type(torch.float32))
            SNOW=torch.mul(PRECIP,(Tm[t,:,:]<parTT).type(torch.float32))

            # Snow
            SNOWPACK=SNOWPACK+SNOW
            melt=parCFMAX*(Tm[t,:,:]-parTT)
            melt=torch.clamp(melt,min=0.0)
            melt=torch.min(melt,SNOWPACK)
            MELTWATER=MELTWATER+melt
            SNOWPACK=SNOWPACK-melt
            refreezing=parCFR*parCFMAX*(parTT-Tm[t,:,:])
            refreezing=torch.clamp(refreezing,min=0.0)
            refreezing=torch.min(refreezing,MELTWATER)
            SNOWPACK=SNOWPACK+refreezing
            MELTWATER=MELTWATER-refreezing
            tosoil=MELTWATER-(parCWH*SNOWPACK)
            tosoil=torch.clamp(tosoil,min=0.0)
            MELTWATER=MELTWATER-tosoil

            # Soil and evaporation
            soil_wetness=(SM/parFC)**parBETA
            soil_wetness=torch.clamp(soil_wetness,min=0.0,max=1.0)
            recharge=(RAIN+tosoil)*soil_wetness

            SM=SM+RAIN+tosoil-recharge
            excess=SM-parFC
            excess=torch.clamp(excess,min=0.0)
            SM=SM-excess
            evapfactor=SM/(parLP*parFC)
            evapfactor=torch.clamp(evapfactor,min=0.0,max=1.0)
            ETact=ETpm[t,:,:]*evapfactor
            ETact=torch.min(SM,ETact)
            SM=torch.clamp(SM-ETact,min=PRECS)  # SM can not be zero for gradient tracking
            ETmu[t,:,:]=ETact

            # Groundwater boxes
            SUZ=SUZ+recharge+excess
            PERC=torch.min(SUZ,parPERC)
            SUZ=SUZ-PERC
            Q0=parK0*torch.clamp(SUZ-parUZL,min=0.0)
            SUZ=SUZ-Q0
            Q1=parK1*SUZ
            SUZ=SUZ-Q1
            SLZ=SLZ+PERC
            Q2=parK2*SLZ
            SLZ=SLZ-Q2
            Qsimmu[t,:,:]=Q0+Q1+Q2

            # save components
            Qsimmu0[t,:,:]=Q0
            Qsimmu1[t,:,:]=Q1
            Qsimmu2[t,:,:]=Q2

        Qsimave0=Qsimmu0.mean(-1,keepdim=True)
        Qsimave1=Qsimmu1.mean(-1,keepdim=True)
        Qsimave2=Qsimmu2.mean(-1,keepdim=True)
        ETave=ETmu.mean(-1,keepdim=True)

        # get the initial component average
        if muwts is None:
            Qsimave=Qsimmu.mean(-1)
        else:
            Qsimave=(Qsimmu*muwts).sum(-1)

        if routOpt is True:  # routing
            if comprout is True:
                # do routing to all the components, reshape the matrix to [Time, gage*multi]
                Qsim=Qsimmu.view(Nstep,Ngrid*mu)
            else:
                # average the components first, then do routing
                Qsim=Qsimave
            # scale learned routing parameter
            tempa=routscaLst[0][0]+rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb=routscaLst[1][0]+rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])
            routa=tempa.repeat(Nstep,1).unsqueeze(-1)
            routb=tempb.repeat(Nstep,1).unsqueeze(-1)
            UH=UH_gamma(routa,routb,lenF=15)  # lenF: folter
            rf=torch.unsqueeze(Qsim,-1).permute([1,2,0])  # dim:gage*var*time
            UH=UH.permute([1,2,0])  # dim: gage*var*time
            Qsrout=UH_conv(rf,UH).permute([2,0,1])

            if comprout is True:  # Qs is [time, [gage*mult], var] now
                Qstemp=Qsrout.view(Nstep,Ngrid,mu)
                if muwts is None:
                    Qs=Qstemp.mean(-1,keepdim=True)
                else:
                    Qs=(Qstemp*muwts).sum(-1,keepdim=True)
            else:
                Qs=Qsrout

        else:  # no routing, output the initial average simulations

            Qs=torch.unsqueeze(Qsimave,-1)  # add a dimension

        if outstate is True:
            return Qs,SNOWPACK,MELTWATER,SM,SUZ,SLZ
        else:
            Qall=torch.cat((Qs,Qsimave0,Qsimave1,Qsimave2,ETave),dim=-1)
            return Qall


class HBVMulET(torch.nn.Module):
    """Multi-component HBV Model PyTorch version"""

    # Add an ET shape parameter; others are the same as class HBVMul()
    # refer HBVMul() for detailed comments

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMulET,self).__init__()

    def forward(self,x,parameters,mu,muwts,rtwts,bufftime=0,outstate=False,routOpt=False,comprout=False):

        PRECS=1e-5

        # Initialization
        if bufftime>0:
            with torch.no_grad():
                xinit=x[0:bufftime,:,:]
                initmodel=HBVMulET()
                Qsinit,SNOWPACK,MELTWATER,SM,SUZ,SLZ=initmodel(xinit,parameters,mu,muwts,rtwts,
                                                               bufftime=0,outstate=True,routOpt=False,comprout=False)
        else:

            # Without buff time, initialize state variables with zeros
            Ngrid=x.shape[1]
            SNOWPACK=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).cuda()
            MELTWATER=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).cuda()
            SM=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).cuda()
            SUZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).cuda()
            SLZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).cuda()
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()

        P=x[bufftime:,:,0]
        Pm=P.unsqueeze(2).repeat(1,1,mu)
        T=x[bufftime:,:,1]
        Tm=T.unsqueeze(2).repeat(1,1,mu)
        ETpot=x[bufftime:,:,2]
        ETpm=ETpot.unsqueeze(2).repeat(1,1,mu)

        ## scale the parameters
        parascaLst=[[1,6],[50,1000],[0.05,0.9],[0.01,0.5],[0.001,0.2],[0.2,1],
                    [0,10],[0,100],[-2.5,2.5],[0.5,10],[0,0.1],[0,0.2],[0.3,5]]
        routscaLst=[[0,2.9],[0,6.5]]

        parBETA=parascaLst[0][0]+parameters[:,0,:]*(parascaLst[0][1]-parascaLst[0][0])
        parFC=parascaLst[1][0]+parameters[:,1,:]*(parascaLst[1][1]-parascaLst[1][0])
        parK0=parascaLst[2][0]+parameters[:,2,:]*(parascaLst[2][1]-parascaLst[2][0])
        parK1=parascaLst[3][0]+parameters[:,3,:]*(parascaLst[3][1]-parascaLst[3][0])
        parK2=parascaLst[4][0]+parameters[:,4,:]*(parascaLst[4][1]-parascaLst[4][0])
        parLP=parascaLst[5][0]+parameters[:,5,:]*(parascaLst[5][1]-parascaLst[5][0])
        parPERC=parascaLst[6][0]+parameters[:,6,:]*(parascaLst[6][1]-parascaLst[6][0])
        parUZL=parascaLst[7][0]+parameters[:,7,:]*(parascaLst[7][1]-parascaLst[7][0])
        parTT=parascaLst[8][0]+parameters[:,8,:]*(parascaLst[8][1]-parascaLst[8][0])
        parCFMAX=parascaLst[9][0]+parameters[:,9,:]*(parascaLst[9][1]-parascaLst[9][0])
        parCFR=parascaLst[10][0]+parameters[:,10,:]*(parascaLst[10][1]-parascaLst[10][0])
        parCWH=parascaLst[11][0]+parameters[:,11,:]*(parascaLst[11][1]-parascaLst[11][0])

        # The added ET parameter
        parBETAET=parascaLst[12][0]+parameters[:,12,:]*(parascaLst[12][1]-parascaLst[12][0])

        Nstep,Ngrid=P.size()

        # Initialize time series of model variables
        Qsimmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).cuda()

        for t in range(Nstep):
            # Separate precipitation into liquid and solid components
            PRECIP=Pm[t,:,:]  # need to check later, seems repeating with line 52
            RAIN=torch.mul(PRECIP,(Tm[t,:,:]>=parTT).type(torch.float32))
            SNOW=torch.mul(PRECIP,(Tm[t,:,:]<parTT).type(torch.float32))

            # Snow
            SNOWPACK=SNOWPACK+SNOW
            melt=parCFMAX*(Tm[t,:,:]-parTT)
            # melt[melt < 0.0] = 0.0
            melt=torch.clamp(melt,min=0.0)
            # melt[melt > SNOWPACK] = SNOWPACK[melt > SNOWPACK]
            melt=torch.min(melt,SNOWPACK)
            MELTWATER=MELTWATER+melt
            SNOWPACK=SNOWPACK-melt
            refreezing=parCFR*parCFMAX*(parTT-Tm[t,:,:])
            # refreezing[refreezing < 0.0] = 0.0
            # refreezing[refreezing > MELTWATER] = MELTWATER[refreezing > MELTWATER]
            refreezing=torch.clamp(refreezing,min=0.0)
            refreezing=torch.min(refreezing,MELTWATER)
            SNOWPACK=SNOWPACK+refreezing
            MELTWATER=MELTWATER-refreezing
            tosoil=MELTWATER-(parCWH*SNOWPACK)
            # tosoil[tosoil < 0.0] = 0.0
            tosoil=torch.clamp(tosoil,min=0.0)
            MELTWATER=MELTWATER-tosoil

            # Soil and evaporation
            soil_wetness=(SM/parFC)**parBETA
            # soil_wetness[soil_wetness < 0.0] = 0.0
            # soil_wetness[soil_wetness > 1.0] = 1.0
            soil_wetness=torch.clamp(soil_wetness,min=0.0,max=1.0)
            recharge=(RAIN+tosoil)*soil_wetness

            SM=SM+RAIN+tosoil-recharge
            excess=SM-parFC
            # excess[excess < 0.0] = 0.0
            excess=torch.clamp(excess,min=0.0)
            SM=SM-excess
            # MODIFY HERE. Add the shape para parBETAET for ET equation
            evapfactor=(SM/(parLP*parFC))**parBETAET
            # evapfactor = SM / (parLP * parFC)
            # evapfactor[evapfactor < 0.0] = 0.0
            # evapfactor[evapfactor > 1.0] = 1.0
            evapfactor=torch.clamp(evapfactor,min=0.0,max=1.0)
            ETact=ETpm[t,:,:]*evapfactor
            ETact=torch.min(SM,ETact)
            SM=torch.clamp(SM-ETact,min=PRECS)  # SM can not be zero for gradient tracking

            # Groundwater boxes
            SUZ=SUZ+recharge+excess
            PERC=torch.min(SUZ,parPERC)
            SUZ=SUZ-PERC
            Q0=parK0*torch.clamp(SUZ-parUZL,min=0.0)
            SUZ=SUZ-Q0
            Q1=parK1*SUZ
            SUZ=SUZ-Q1
            SLZ=SLZ+PERC
            Q2=parK2*SLZ
            SLZ=SLZ-Q2
            Qsimmu[t,:,:]=Q0+Q1+Q2

        # get the initial average
        if muwts is None:
            Qsimave=Qsimmu.mean(-1)
        else:
            Qsimave=(Qsimmu*muwts).sum(-1)

        if routOpt is True:  # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim=Qsimmu.view(Nstep,Ngrid*mu)
            else:
                # average the components, then do routing
                Qsim=Qsimave

            tempa=routscaLst[0][0]+rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb=routscaLst[1][0]+rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])
            routa=tempa.repeat(Nstep,1).unsqueeze(-1)
            routb=tempb.repeat(Nstep,1).unsqueeze(-1)
            UH=UH_gamma(routa,routb,lenF=15)  # lenF: folter
            rf=torch.unsqueeze(Qsim,-1).permute([1,2,0])  # dim:gage*var*time
            UH=UH.permute([1,2,0])  # dim: gage*var*time
            Qsrout=UH_conv(rf,UH).permute([2,0,1])

            if comprout is True:  # Qs is [time, [gage*mult], var] now
                Qstemp=Qsrout.view(Nstep,Ngrid,mu)
                if muwts is None:
                    Qs=Qstemp.mean(-1,keepdim=True)
                else:
                    Qs=(Qstemp*muwts).sum(-1,keepdim=True)
            else:
                Qs=Qsrout

        else:  # no routing, output the initial average simulations

            Qs=torch.unsqueeze(Qsimave,-1)  # add a dimension

        if outstate is True:
            return Qs,SNOWPACK,MELTWATER,SM,SUZ,SLZ
        else:
            return Qs  # total streamflow


class MultiInv_HBVModel(torch.nn.Module):
    # class for dPL + HBV with multiple components and static parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,pcorr=None):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVModel,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        if pcorr is None:
            ntp=nhbvpm+nroutpm+nwtspm
        else:
            ntp=nhbvpm+nroutpm+nwtspm+1  # 1 for potential precipitation correction
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.HBV=HBVMul()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.pcorr=pcorr

    def forward(self,x,z,doDropMC=False):
        Gen=self.lstminv(z)
        Params0=Gen[-1,:,:]  # the last time step as learned parameters
        ngage=Params0.shape[0]
        # print(Params0)
        hbvpara0=Params0[:,0:self.nhbvpm]
        hbvpara=torch.sigmoid(hbvpara0).view(ngage,self.nfea,self.nmul)
        routpara0=Params0[:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # dim: [Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:  # if do routing for each component
            routpara=torch.sigmoid(routpara0)
        else:
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)
        if self.nwtspm==0:  # 0: simple average instead of weighted average for all components
            wts=None
        else:
            wtspara=Params0[:,self.nhbvpm+self.nroutpm:self.nhbvpm+self.nroutpm+self.nwtspm]
            wts=F.softmax(wtspara,dim=-1)
        if self.pcorr is None:
            corrpara=None
        else:
            corrpara0=Params0[:,self.nhbvpm+self.nroutpm+self.nwtspm:self.nhbvpm+self.nroutpm+self.nwtspm+1]
            corrpara=torch.sigmoid(corrpara0)
        out=self.HBV(x,parameters=hbvpara,mu=self.nmul,muwts=wts,rtwts=routpara,bufftime=self.inittime,
                     routOpt=self.routOpt,comprout=self.comprout,corrwts=corrpara,pcorr=self.pcorr)  # HBV forward
        return out


class HBVMulTD(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version implemented by Dapeng Feng"""

    # we suggest you read the class HBVMul() with static parameters first

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMulTD,self).__init__()

    def forward(self,x,parameters,staind,tdlst,mu,muwts,rtwts,bufftime=0,outstate=False,routOpt=False,
                comprout=False,dydrop=False):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     x: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2]
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     dydrop: the possibility to drop a dynamic para to static to reduce potential overfitting
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET

        PRECS=1e-5

        # Initialization
        if bufftime>0:
            with torch.no_grad():
                xinit=x[0:bufftime,:,:]
                initmodel=HBVMul()
                buffpara=parameters[bufftime-1,:,:,:]
                Qsinit,SNOWPACK,MELTWATER,SM,SUZ,SLZ=initmodel(xinit,buffpara,mu,muwts,rtwts,
                                                               bufftime=0,outstate=True,routOpt=False,comprout=False)
        else:

            # Without buff time, initialize state variables with zeros
            Ngrid=x.shape[1]
            SNOWPACK=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).cuda()
            MELTWATER=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).cuda()
            SM=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).cuda()
            SUZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).cuda()
            SLZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).cuda()
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()

        P=x[bufftime:,:,0]
        Pm=P.unsqueeze(2).repeat(1,1,mu)
        T=x[bufftime:,:,1]
        Tm=T.unsqueeze(2).repeat(1,1,mu)
        ETpot=x[bufftime:,:,2]
        ETpm=ETpot.unsqueeze(2).repeat(1,1,mu)
        parAll=parameters[bufftime:,:,:,:]
        parAllTrans=torch.zeros_like(parAll)

        ## scale the parameters
        hbvscaLst=[[1,6],[50,1000],[0.05,0.9],[0.01,0.5],[0.001,0.2],[0.2,1],
                   [0,10],[0,100],[-2.5,2.5],[0.5,10],[0,0.1],[0,0.2]]
        routscaLst=[[0,2.9],[0,6.5]]

        for ip in range(len(hbvscaLst)):  # not include routing. Scaling the parameters using loop
            parAllTrans[:,:,ip,:]=hbvscaLst[ip][0]+parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        Nstep,Ngrid=P.size()

        # deal with the dynamic parameters and dropout
        parstaFull=parAllTrans[staind,:,:,:].unsqueeze(0).repeat([Nstep,1,1,1])  # static matrix
        parhbvFull=torch.clone(parstaFull)
        # create probability mask for each parameter on the basin dimension to apply dropout
        pmat=torch.ones([1,Ngrid,1])*dydrop
        for ix in tdlst:
            staPar=parstaFull[:,:,ix-1,:]
            dynPar=parAllTrans[:,:,ix-1,:]
            drmask=torch.bernoulli(pmat).detach_().cuda()  # to drop some dynamic parameters as static
            comPar=dynPar*(1-drmask)+staPar*drmask
            parhbvFull[:,:,ix-1,:]=comPar

        # Initialize time series of model variables
        Qsimmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).cuda()
        ETmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).cuda()

        # Output the box components of Q
        Qsimmu0=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).cuda()
        Qsimmu1=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).cuda()
        Qsimmu2=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).cuda()

        for t in range(Nstep):
            paraLst=[]
            for ip in range(len(hbvscaLst)):  # unpack HBV parameters
                paraLst.append(parhbvFull[t,:,ip,:])

            parBETA,parFC,parK0,parK1,parK2,parLP,parPERC,parUZL,parTT,parCFMAX,parCFR,parCWH=paraLst

            # Separate precipitation into liquid and solid components
            PRECIP=Pm[t,:,:]
            RAIN=torch.mul(PRECIP,(Tm[t,:,:]>=parTT).type(torch.float32))
            SNOW=torch.mul(PRECIP,(Tm[t,:,:]<parTT).type(torch.float32))

            # Snow
            SNOWPACK=SNOWPACK+SNOW
            melt=parCFMAX*(Tm[t,:,:]-parTT)
            # melt[melt < 0.0] = 0.0
            melt=torch.clamp(melt,min=0.0)
            # melt[melt > SNOWPACK] = SNOWPACK[melt > SNOWPACK]
            melt=torch.min(melt,SNOWPACK)
            MELTWATER=MELTWATER+melt
            SNOWPACK=SNOWPACK-melt
            refreezing=parCFR*parCFMAX*(parTT-Tm[t,:,:])
            # refreezing[refreezing < 0.0] = 0.0
            # refreezing[refreezing > MELTWATER] = MELTWATER[refreezing > MELTWATER]
            refreezing=torch.clamp(refreezing,min=0.0)
            refreezing=torch.min(refreezing,MELTWATER)
            SNOWPACK=SNOWPACK+refreezing
            MELTWATER=MELTWATER-refreezing
            tosoil=MELTWATER-(parCWH*SNOWPACK)
            # tosoil[tosoil < 0.0] = 0.0
            tosoil=torch.clamp(tosoil,min=0.0)
            MELTWATER=MELTWATER-tosoil

            # Soil and evaporation
            soil_wetness=(SM/parFC)**parBETA
            # soil_wetness[soil_wetness < 0.0] = 0.0
            # soil_wetness[soil_wetness > 1.0] = 1.0
            soil_wetness=torch.clamp(soil_wetness,min=0.0,max=1.0)
            recharge=(RAIN+tosoil)*soil_wetness

            SM=SM+RAIN+tosoil-recharge
            excess=SM-parFC
            # excess[excess < 0.0] = 0.0
            excess=torch.clamp(excess,min=0.0)
            SM=SM-excess
            evapfactor=SM/(parLP*parFC)
            evapfactor=torch.clamp(evapfactor,min=0.0,max=1.0)
            ETact=ETpm[t,:,:]*evapfactor
            ETact=torch.min(SM,ETact)
            SM=torch.clamp(SM-ETact,min=PRECS)  # SM can not be zero for gradient tracking

            # Groundwater boxes
            SUZ=SUZ+recharge+excess
            PERC=torch.min(SUZ,parPERC)
            SUZ=SUZ-PERC
            Q0=parK0*torch.clamp(SUZ-parUZL,min=0.0)
            SUZ=SUZ-Q0
            Q1=parK1*SUZ
            SUZ=SUZ-Q1
            SLZ=SLZ+PERC
            Q2=parK2*SLZ
            SLZ=SLZ-Q2
            Qsimmu[t,:,:]=Q0+Q1+Q2

            # save components
            Qsimmu0[t,:,:]=Q0
            Qsimmu1[t,:,:]=Q1
            Qsimmu2[t,:,:]=Q2
            ETmu[t,:,:]=ETact

        Qsimave0=Qsimmu0.mean(-1,keepdim=True)
        Qsimave1=Qsimmu1.mean(-1,keepdim=True)
        Qsimave2=Qsimmu2.mean(-1,keepdim=True)
        ETave=ETmu.mean(-1,keepdim=True)

        # get the initial average
        if muwts is None:
            Qsimave=Qsimmu.mean(-1)
        else:
            Qsimave=(Qsimmu*muwts).sum(-1)

        if routOpt is True:  # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim=Qsimmu.view(Nstep,Ngrid*mu)
            else:
                # average the components, then do routing
                Qsim=Qsimave

            tempa=routscaLst[0][0]+rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb=routscaLst[1][0]+rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])
            routa=tempa.repeat(Nstep,1).unsqueeze(-1)
            routb=tempb.repeat(Nstep,1).unsqueeze(-1)
            UH=UH_gamma(routa,routb,lenF=15)  # lenF: folter
            rf=torch.unsqueeze(Qsim,-1).permute([1,2,0])  # dim:gage*var*time
            UH=UH.permute([1,2,0])  # dim: gage*var*time
            Qsrout=UH_conv(rf,UH).permute([2,0,1])

            if comprout is True:  # Qs is [time, [gage*mult], var] now
                Qstemp=Qsrout.view(Nstep,Ngrid,mu)
                if muwts is None:
                    Qs=Qstemp.mean(-1,keepdim=True)
                else:
                    Qs=(Qstemp*muwts).sum(-1,keepdim=True)
            else:
                Qs=Qsrout

        else:  # no routing, output the primary average simulations

            Qs=torch.unsqueeze(Qsimave,-1)  # add a dimension

        if outstate is True:
            return Qs,SNOWPACK,MELTWATER,SM,SUZ,SLZ
        else:
            # return Qs
            Qall=torch.cat((Qs,Qsimave0,Qsimave1,Qsimave2,ETave),dim=-1)
            return Qall


class HBVMulTDET(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""

    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMulTDET,self).__init__()

    def forward(self,x,parameters,staind,tdlst,mu,muwts,rtwts,bufftime=0,outstate=False,routOpt=False,
                comprout=False,dydrop=False):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     X: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2]
        #     staind:use which time step from the learned para time series for static parameters
        #     tdlst: the index list of hbv parameters set as dynamic
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     dydrop: the possibility to drop a dynamic para to static to reduce potential overfitting
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET

        PRECS=1e-5  # keep the numerical calculation stable

        # Initialization to warm-up states
        if bufftime>0:
            with torch.no_grad():
                xinit=x[0:bufftime,:,:]
                initmodel=HBVMulET()
                buffpara=parameters[bufftime-1,:,:,:]
                Qsinit,SNOWPACK,MELTWATER,SM,SUZ,SLZ=initmodel(xinit,buffpara,mu,muwts,rtwts,
                                                               bufftime=0,outstate=True,routOpt=False,comprout=False)
        else:

            # Without warm-up bufftime=0, initialize state variables with zeros
            Ngrid=x.shape[1]
            SNOWPACK=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).cuda()
            MELTWATER=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).cuda()
            SM=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).cuda()
            SUZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).cuda()
            SLZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).cuda()
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()

        P=x[bufftime:,:,0]
        Pm=P.unsqueeze(2).repeat(1,1,mu)  # precip
        T=x[bufftime:,:,1]
        Tm=T.unsqueeze(2).repeat(1,1,mu)  # temperature
        ETpot=x[bufftime:,:,2]
        ETpm=ETpot.unsqueeze(2).repeat(1,1,mu)  # potential ET
        parAll=parameters[bufftime:,:,:,:]
        parAllTrans=torch.zeros_like(parAll)

        ## scale the parameters to real values from [0,1]
        hbvscaLst=[[1,6],[50,1000],[0.05,0.9],[0.01,0.5],[0.001,0.2],[0.2,1],
                   [0,10],[0,100],[-2.5,2.5],[0.5,10],[0,0.1],[0,0.2],[0.3,5]]  # HBV para
        routscaLst=[[0,2.9],[0,6.5]]  # routing para

        for ip in range(len(hbvscaLst)):  # not include routing. Scaling the parameters
            parAllTrans[:,:,ip,:]=hbvscaLst[ip][0]+parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        Nstep,Ngrid=P.size()

        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        parstaFull=parAllTrans[staind,:,:,:].unsqueeze(0).repeat([Nstep,1,1,1])  # static para matrix
        parhbvFull=torch.clone(parstaFull)
        # create probability mask for each parameter on the basin dimension
        pmat=torch.ones([1,Ngrid,1])*dydrop
        for ix in tdlst:
            staPar=parstaFull[:,:,ix-1,:]
            dynPar=parAllTrans[:,:,ix-1,:]
            drmask=torch.bernoulli(pmat).detach_().cuda()  # to drop dynamic parameters as static in some basins
            comPar=dynPar*(1-drmask)+staPar*drmask
            parhbvFull[:,:,ix-1,:]=comPar

        # Initialize time series of model variables to save results
        Qsimmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).cuda()
        ETmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).cuda()

        # Output the box components of Q
        Qsimmu0=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).cuda()
        Qsimmu1=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).cuda()
        Qsimmu2=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).cuda()

        # # Not used. Logging the state variables for debug.
        # # SMlog = np.zeros(P.size())
        # logSM = np.zeros(P.size())
        # logPS = np.zeros(P.size())
        # logswet = np.zeros(P.size())
        # logRE = np.zeros(P.size())

        for t in range(Nstep):
            paraLst=[]
            for ip in range(len(hbvscaLst)):  # unpack HBV parameters
                paraLst.append(parhbvFull[t,:,ip,:])

            parBETA,parFC,parK0,parK1,parK2,parLP,parPERC,parUZL,parTT,parCFMAX,parCFR,parCWH,parBETAET=paraLst
            # Separate precipitation into liquid and solid components
            PRECIP=Pm[t,:,:]
            RAIN=torch.mul(PRECIP,(Tm[t,:,:]>=parTT).type(torch.float32))
            SNOW=torch.mul(PRECIP,(Tm[t,:,:]<parTT).type(torch.float32))

            # Snow process
            SNOWPACK=SNOWPACK+SNOW
            melt=parCFMAX*(Tm[t,:,:]-parTT)
            melt=torch.clamp(melt,min=0.0)
            melt=torch.min(melt,SNOWPACK)
            MELTWATER=MELTWATER+melt
            SNOWPACK=SNOWPACK-melt
            refreezing=parCFR*parCFMAX*(parTT-Tm[t,:,:])
            refreezing=torch.clamp(refreezing,min=0.0)
            refreezing=torch.min(refreezing,MELTWATER)
            SNOWPACK=SNOWPACK+refreezing
            MELTWATER=MELTWATER-refreezing
            tosoil=MELTWATER-(parCWH*SNOWPACK)
            tosoil=torch.clamp(tosoil,min=0.0)
            MELTWATER=MELTWATER-tosoil

            # Soil and evaporation
            soil_wetness=(SM/parFC)**parBETA
            soil_wetness=torch.clamp(soil_wetness,min=0.0,max=1.0)
            recharge=(RAIN+tosoil)*soil_wetness

            # Not used, logging states for checking
            # logSM[t,:] = SM.detach().cpu().numpy()
            # logPS[t,:] = (RAIN + tosoil).detach().cpu().numpy()
            # logswet[t,:] = (SM / parFC).detach().cpu().numpy()
            # logRE[t, :] = recharge.detach().cpu().numpy()

            SM=SM+RAIN+tosoil-recharge
            excess=SM-parFC
            excess=torch.clamp(excess,min=0.0)
            SM=SM-excess
            # MODIFY here. Different as class HBVMulT(). Add a ET shape parameter parBETAET
            evapfactor=(SM/(parLP*parFC))**parBETAET
            evapfactor=torch.clamp(evapfactor,min=0.0,max=1.0)
            ETact=ETpm[t,:,:]*evapfactor
            ETact=torch.min(SM,ETact)
            SM=torch.clamp(SM-ETact,min=PRECS)  # SM can not be zero for gradient tracking

            # Groundwater boxes
            SUZ=SUZ+recharge+excess
            PERC=torch.min(SUZ,parPERC)
            SUZ=SUZ-PERC
            Q0=parK0*torch.clamp(SUZ-parUZL,min=0.0)
            SUZ=SUZ-Q0
            Q1=parK1*SUZ
            SUZ=SUZ-Q1
            SLZ=SLZ+PERC
            Q2=parK2*SLZ
            SLZ=SLZ-Q2
            Qsimmu[t,:,:]=Q0+Q1+Q2

            # save components for Q
            Qsimmu0[t,:,:]=Q0
            Qsimmu1[t,:,:]=Q1
            Qsimmu2[t,:,:]=Q2
            ETmu[t,:,:]=ETact

            # Not used, for debug state variables
            # SMlog[t,:, :] = SM.detach().cpu().numpy()
            # SUZlog[t,:,:] = SUZ.detach().cpu().numpy()
            # SLZlog[t,:,:] = SLZ.detach().cpu().numpy()

        Qsimave0=Qsimmu0.mean(-1,keepdim=True)
        Qsimave1=Qsimmu1.mean(-1,keepdim=True)
        Qsimave2=Qsimmu2.mean(-1,keepdim=True)
        ETave=ETmu.mean(-1,keepdim=True)

        # get the initial average
        if muwts is None:  # simple average
            Qsimave=Qsimmu.mean(-1)
        else:  # weighted average using learned weights
            Qsimave=(Qsimmu*muwts).sum(-1)

        if routOpt is True:  # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim=Qsimmu.view(Nstep,Ngrid*mu)
            else:
                # average the components, then do routing
                Qsim=Qsimave

            # scale two routing parameters
            tempa=routscaLst[0][0]+rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb=routscaLst[1][0]+rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])
            routa=tempa.repeat(Nstep,1).unsqueeze(-1)
            routb=tempb.repeat(Nstep,1).unsqueeze(-1)
            UH=UH_gamma(routa,routb,lenF=15)  # lenF: folter
            rf=torch.unsqueeze(Qsim,-1).permute([1,2,0])  # dim:gage*var*time
            UH=UH.permute([1,2,0])  # dim: gage*var*time
            Qsrout=UH_conv(rf,UH).permute([2,0,1])

            if comprout is True:  # Qs is [time, [gage*mult], var] now
                Qstemp=Qsrout.view(Nstep,Ngrid,mu)
                if muwts is None:
                    Qs=Qstemp.mean(-1,keepdim=True)
                else:
                    Qs=(Qstemp*muwts).sum(-1,keepdim=True)
            else:
                Qs=Qsrout

        else:  # no routing, output the initial average simulations
            Qs=torch.unsqueeze(Qsimave,-1)  # add a dimension

        if outstate is True:  # output states
            return Qs,SNOWPACK,MELTWATER,SM,SUZ,SLZ
        else:
            # return Qs
            Qall=torch.cat((Qs,Qsimave0,Qsimave1,Qsimave2,ETave),dim=-1)
            return Qall


class HBVMulTDET_all(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""

    # all are dynamic parameters
    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMulTDET_all,self).__init__()

    def forward(self,x,parameters,staind,tdlst,mu,muwts,rtwts,bufftime=0,outstate=False,routOpt=False,
                comprout=False,dydrop=False):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     X: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2]
        #     staind:use which time step from the learned para time series for static parameters
        #     tdlst: the index list of hbv parameters set as dynamic
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     dydrop: the possibility to drop a dynamic para to static to reduce potential overfitting
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET

        PRECS=1e-5  # keep the numerical calculation stable

        # Initialization to warm-up states
        if bufftime>0:
            with torch.no_grad():
                xinit=x[0:bufftime,:,:]
                initmodel=HBVMulET()
                buffpara=parameters[bufftime-1,:,:,:]
                Qsinit,SNOWPACK,MELTWATER,SM,SUZ,SLZ=initmodel(xinit,buffpara,mu,muwts,rtwts,
                                                               bufftime=0,outstate=True,routOpt=False,comprout=False)
        else:

            # Without warm-up bufftime=0, initialize state variables with zeros
            Ngrid=x.shape[1]
            SNOWPACK=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).cuda()
            MELTWATER=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).cuda()
            SM=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).cuda()
            SUZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).cuda()
            SLZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).cuda()
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()

        P=x[bufftime:,:,0]
        Pm=P.unsqueeze(2).repeat(1,1,mu)  # precip
        T=x[bufftime:,:,1]
        Tm=T.unsqueeze(2).repeat(1,1,mu)  # temperature
        ETpot=x[bufftime:,:,2]
        ETpm=ETpot.unsqueeze(2).repeat(1,1,mu)  # potential ET
        parAll=parameters[bufftime:,:,:,:]
        parAllTrans=torch.zeros_like(parAll)

        ## scale the parameters to real values from [0,1]
        hbvscaLst=[[1,6],[50,1000],[0.05,0.9],[0.01,0.5],[0.001,0.2],[0.2,1],
                   [0,10],[0,100],[-2.5,2.5],[0.5,10],[0,0.1],[0,0.2],[0.3,5]]  # HBV para
        routscaLst=[[0,2.9],[0,6.5]]  # routing para

        for ip in range(len(hbvscaLst)):  # not include routing. Scaling the parameters
            parAllTrans[:,:,ip,:]=hbvscaLst[ip][0]+parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        Nstep,Ngrid=P.size()

        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        parstaFull=parAllTrans[staind,:,:,:].unsqueeze(0).repeat([Nstep,1,1,1])  # static para matrix
        parhbvFull=torch.clone(parstaFull)
        # create probability mask for each parameter on the basin dimension
        pmat=torch.ones([1,Ngrid,1])*dydrop
        for ix in [1,2,3,4,5,6,7,8,9,10,11,12,13]:
            staPar=parstaFull[:,:,ix-1,:]
            dynPar=parAllTrans[:,:,ix-1,:]
            drmask=torch.bernoulli(pmat).detach_().cuda()  # to drop dynamic parameters as static in some basins
            comPar=dynPar*(1-drmask)+staPar*drmask
            parhbvFull[:,:,ix-1,:]=comPar

        # Initialize time series of model variables to save results
        Qsimmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).cuda()
        ETmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).cuda()

        # Output the box components of Q
        Qsimmu0=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).cuda()
        Qsimmu1=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).cuda()
        Qsimmu2=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).cuda()

        # # Not used. Logging the state variables for debug.
        # # SMlog = np.zeros(P.size())
        # logSM = np.zeros(P.size())
        # logPS = np.zeros(P.size())
        # logswet = np.zeros(P.size())
        # logRE = np.zeros(P.size())

        for t in range(Nstep):
            paraLst=[]
            for ip in range(len(hbvscaLst)):  # unpack HBV parameters
                paraLst.append(parhbvFull[t,:,ip,:])

            parBETA,parFC,parK0,parK1,parK2,parLP,parPERC,parUZL,parTT,parCFMAX,parCFR,parCWH,parBETAET=paraLst
            # Separate precipitation into liquid and solid components
            PRECIP=Pm[t,:,:]
            RAIN=torch.mul(PRECIP,(Tm[t,:,:]>=parTT).type(torch.float32))
            SNOW=torch.mul(PRECIP,(Tm[t,:,:]<parTT).type(torch.float32))

            # Snow process
            SNOWPACK=SNOWPACK+SNOW
            melt=parCFMAX*(Tm[t,:,:]-parTT)
            melt=torch.clamp(melt,min=0.0)
            melt=torch.min(melt,SNOWPACK)
            MELTWATER=MELTWATER+melt
            SNOWPACK=SNOWPACK-melt
            refreezing=parCFR*parCFMAX*(parTT-Tm[t,:,:])
            refreezing=torch.clamp(refreezing,min=0.0)
            refreezing=torch.min(refreezing,MELTWATER)
            SNOWPACK=SNOWPACK+refreezing
            MELTWATER=MELTWATER-refreezing
            tosoil=MELTWATER-(parCWH*SNOWPACK)
            tosoil=torch.clamp(tosoil,min=0.0)
            MELTWATER=MELTWATER-tosoil

            # Soil and evaporation
            soil_wetness=(SM/parFC)**parBETA
            soil_wetness=torch.clamp(soil_wetness,min=0.0,max=1.0)
            recharge=(RAIN+tosoil)*soil_wetness

            # Not used, logging states for checking
            # logSM[t,:] = SM.detach().cpu().numpy()
            # logPS[t,:] = (RAIN + tosoil).detach().cpu().numpy()
            # logswet[t,:] = (SM / parFC).detach().cpu().numpy()
            # logRE[t, :] = recharge.detach().cpu().numpy()

            SM=SM+RAIN+tosoil-recharge
            excess=SM-parFC
            excess=torch.clamp(excess,min=0.0)
            SM=SM-excess
            # MODIFY here. Different as class HBVMulT(). Add a ET shape parameter parBETAET
            evapfactor=(SM/(parLP*parFC))**parBETAET
            evapfactor=torch.clamp(evapfactor,min=0.0,max=1.0)
            ETact=ETpm[t,:,:]*evapfactor
            ETact=torch.min(SM,ETact)
            SM=torch.clamp(SM-ETact,min=PRECS)  # SM can not be zero for gradient tracking

            # Groundwater boxes
            SUZ=SUZ+recharge+excess
            PERC=torch.min(SUZ,parPERC)
            SUZ=SUZ-PERC
            Q0=parK0*torch.clamp(SUZ-parUZL,min=0.0)
            SUZ=SUZ-Q0
            Q1=parK1*SUZ
            SUZ=SUZ-Q1
            SLZ=SLZ+PERC
            Q2=parK2*SLZ
            SLZ=SLZ-Q2
            Qsimmu[t,:,:]=Q0+Q1+Q2

            # save components for Q
            Qsimmu0[t,:,:]=Q0
            Qsimmu1[t,:,:]=Q1
            Qsimmu2[t,:,:]=Q2
            ETmu[t,:,:]=ETact

            # Not used, for debug state variables
            # SMlog[t,:, :] = SM.detach().cpu().numpy()
            # SUZlog[t,:,:] = SUZ.detach().cpu().numpy()
            # SLZlog[t,:,:] = SLZ.detach().cpu().numpy()

        Qsimave0=Qsimmu0.mean(-1,keepdim=True)
        Qsimave1=Qsimmu1.mean(-1,keepdim=True)
        Qsimave2=Qsimmu2.mean(-1,keepdim=True)
        ETave=ETmu.mean(-1,keepdim=True)

        # get the initial average
        if muwts is None:  # simple average
            Qsimave=Qsimmu.mean(-1)
        else:  # weighted average using learned weights
            Qsimave=(Qsimmu*muwts).sum(-1)

        if routOpt is True:  # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim=Qsimmu.view(Nstep,Ngrid*mu)
            else:
                # average the components, then do routing
                Qsim=Qsimave

            # scale two routing parameters
            tempa=routscaLst[0][0]+rtwts[:,:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb=routscaLst[1][0]+rtwts[:,:,1]*(routscaLst[1][1]-routscaLst[1][0])
            routa=tempa.unsqueeze(-1)
            routb=tempb.unsqueeze(-1)
            # routa = tempa.repeat(Nstep, 1).unsqueeze(-1)
            # routb = tempb.repeat(Nstep, 1).unsqueeze(-1)
            UH=UH_gamma(routa,routb,lenF=15)  # lenF: folter
            rf=torch.unsqueeze(Qsim,-1).permute([1,2,0])  # dim:gage*var*time
            UH=UH.permute([1,2,0])  # dim: gage*var*time
            Qsrout=UH_conv(rf,UH).permute([2,0,1])

            if comprout is True:  # Qs is [time, [gage*mult], var] now
                Qstemp=Qsrout.view(Nstep,Ngrid,mu)
                if muwts is None:
                    Qs=Qstemp.mean(-1,keepdim=True)
                else:
                    Qs=(Qstemp*muwts).sum(-1,keepdim=True)
            else:
                Qs=Qsrout

        else:  # no routing, output the initial average simulations
            Qs=torch.unsqueeze(Qsimave,-1)  # add a dimension

        if outstate is True:  # output states
            return Qs,SNOWPACK,MELTWATER,SM,SUZ,SLZ
        else:
            # return Qs
            Qall=torch.cat((Qs,Qsimave0,Qsimave1,Qsimave2,ETave),dim=-1)
            return Qall


class HBVMulTDET_all_noRT(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""

    # all are dynamic parameters
    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMulTDET_all_noRT,self).__init__()

    def forward(self,x,parameters,staind,tdlst,mu,muwts,rtwts,bufftime=0,outstate=False,routOpt=False,
                comprout=False,dydrop=False):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     X: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2]
        #     staind:use which time step from the learned para time series for static parameters
        #     tdlst: the index list of hbv parameters set as dynamic
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     dydrop: the possibility to drop a dynamic para to static to reduce potential overfitting
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET

        PRECS=1e-5  # keep the numerical calculation stable

        # Initialization to warm-up states
        if bufftime>0:
            with torch.no_grad():
                xinit=x[0:bufftime,:,:]
                initmodel=HBVMulET()
                buffpara=parameters[bufftime-1,:,:,:]
                Qsinit,SNOWPACK,MELTWATER,SM,SUZ,SLZ=initmodel(xinit,buffpara,mu,muwts,rtwts,
                                                               bufftime=0,outstate=True,routOpt=False,comprout=False)
        else:

            # Without warm-up bufftime=0, initialize state variables with zeros
            Ngrid=x.shape[1]
            SNOWPACK=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).cuda()
            MELTWATER=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).cuda()
            SM=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).cuda()
            SUZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).cuda()
            SLZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).cuda()
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).cuda()

        P=x[bufftime:,:,0]
        Pm=P.unsqueeze(2).repeat(1,1,mu)  # precip
        T=x[bufftime:,:,1]
        Tm=T.unsqueeze(2).repeat(1,1,mu)  # temperature
        ETpot=x[bufftime:,:,2]
        ETpm=ETpot.unsqueeze(2).repeat(1,1,mu)  # potential ET
        parAll=parameters[bufftime:,:,:,:]
        parAllTrans=torch.zeros_like(parAll)

        ## scale the parameters to real values from [0,1]
        hbvscaLst=[[1,6],[50,1000],[0.05,0.9],[0.01,0.5],[0.001,0.2],[0.2,1],
                   [0,10],[0,100],[-2.5,2.5],[0.5,10],[0,0.1],[0,0.2],[0.3,5]]  # HBV para
        routscaLst=[[0,2.9],[0,6.5]]  # routing para

        for ip in range(len(hbvscaLst)):  # not include routing. Scaling the parameters
            parAllTrans[:,:,ip,:]=hbvscaLst[ip][0]+parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        Nstep,Ngrid=P.size()

        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        parstaFull=parAllTrans[staind,:,:,:].unsqueeze(0).repeat([Nstep,1,1,1])  # static para matrix
        parhbvFull=torch.clone(parstaFull)
        # create probability mask for each parameter on the basin dimension
        pmat=torch.ones([1,Ngrid,1])*dydrop
        for ix in [1,2,3,4,5,6,7,8,9,10,11,12,13]:
            staPar=parstaFull[:,:,ix-1,:]
            dynPar=parAllTrans[:,:,ix-1,:]
            drmask=torch.bernoulli(pmat).detach_().cuda()  # to drop dynamic parameters as static in some basins
            comPar=dynPar*(1-drmask)+staPar*drmask
            parhbvFull[:,:,ix-1,:]=comPar

        # Initialize time series of model variables to save results
        Qsimmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).cuda()
        ETmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).cuda()

        # Output the box components of Q
        Qsimmu0=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).cuda()
        Qsimmu1=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).cuda()
        Qsimmu2=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).cuda()

        # # Not used. Logging the state variables for debug.
        # # SMlog = np.zeros(P.size())
        # logSM = np.zeros(P.size())
        # logPS = np.zeros(P.size())
        # logswet = np.zeros(P.size())
        # logRE = np.zeros(P.size())

        for t in range(Nstep):
            paraLst=[]
            for ip in range(len(hbvscaLst)):  # unpack HBV parameters
                paraLst.append(parhbvFull[t,:,ip,:])

            parBETA,parFC,parK0,parK1,parK2,parLP,parPERC,parUZL,parTT,parCFMAX,parCFR,parCWH,parBETAET=paraLst
            # Separate precipitation into liquid and solid components
            PRECIP=Pm[t,:,:]
            RAIN=torch.mul(PRECIP,(Tm[t,:,:]>=parTT).type(torch.float32))
            SNOW=torch.mul(PRECIP,(Tm[t,:,:]<parTT).type(torch.float32))

            # Snow process
            SNOWPACK=SNOWPACK+SNOW
            melt=parCFMAX*(Tm[t,:,:]-parTT)
            melt=torch.clamp(melt,min=0.0)
            melt=torch.min(melt,SNOWPACK)
            MELTWATER=MELTWATER+melt
            SNOWPACK=SNOWPACK-melt
            refreezing=parCFR*parCFMAX*(parTT-Tm[t,:,:])
            refreezing=torch.clamp(refreezing,min=0.0)
            refreezing=torch.min(refreezing,MELTWATER)
            SNOWPACK=SNOWPACK+refreezing
            MELTWATER=MELTWATER-refreezing
            tosoil=MELTWATER-(parCWH*SNOWPACK)
            tosoil=torch.clamp(tosoil,min=0.0)
            MELTWATER=MELTWATER-tosoil

            # Soil and evaporation
            soil_wetness=(SM/parFC)**parBETA
            soil_wetness=torch.clamp(soil_wetness,min=0.0,max=1.0)
            recharge=(RAIN+tosoil)*soil_wetness

            # Not used, logging states for checking
            # logSM[t,:] = SM.detach().cpu().numpy()
            # logPS[t,:] = (RAIN + tosoil).detach().cpu().numpy()
            # logswet[t,:] = (SM / parFC).detach().cpu().numpy()
            # logRE[t, :] = recharge.detach().cpu().numpy()

            SM=SM+RAIN+tosoil-recharge
            excess=SM-parFC
            excess=torch.clamp(excess,min=0.0)
            SM=SM-excess
            # MODIFY here. Different as class HBVMulT(). Add a ET shape parameter parBETAET
            evapfactor=(SM/(parLP*parFC))**parBETAET
            evapfactor=torch.clamp(evapfactor,min=0.0,max=1.0)
            ETact=ETpm[t,:,:]*evapfactor
            ETact=torch.min(SM,ETact)
            SM=torch.clamp(SM-ETact,min=PRECS)  # SM can not be zero for gradient tracking

            # Groundwater boxes
            SUZ=SUZ+recharge+excess
            PERC=torch.min(SUZ,parPERC)
            SUZ=SUZ-PERC
            Q0=parK0*torch.clamp(SUZ-parUZL,min=0.0)
            SUZ=SUZ-Q0
            Q1=parK1*SUZ
            SUZ=SUZ-Q1
            SLZ=SLZ+PERC
            Q2=parK2*SLZ
            SLZ=SLZ-Q2
            Qsimmu[t,:,:]=Q0+Q1+Q2

            # save components for Q
            Qsimmu0[t,:,:]=Q0
            Qsimmu1[t,:,:]=Q1
            Qsimmu2[t,:,:]=Q2
            ETmu[t,:,:]=ETact

            # Not used, for debug state variables
            # SMlog[t,:, :] = SM.detach().cpu().numpy()
            # SUZlog[t,:,:] = SUZ.detach().cpu().numpy()
            # SLZlog[t,:,:] = SLZ.detach().cpu().numpy()

        Qsimave0=Qsimmu0.mean(-1,keepdim=True)
        Qsimave1=Qsimmu1.mean(-1,keepdim=True)
        Qsimave2=Qsimmu2.mean(-1,keepdim=True)
        ETave=ETmu.mean(-1,keepdim=True)

        # get the initial average
        if muwts is None:  # simple average
            Qsimave=Qsimmu.mean(-1)
        else:  # weighted average using learned weights
            Qsimave=(Qsimmu*muwts).sum(-1)

        if routOpt is True:  # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim=Qsimmu.view(Nstep,Ngrid*mu)
            else:
                # average the components, then do routing
                Qsim=Qsimave

            # scale two routing parameters
            # tempa = routscaLst[0][0] + rtwts[:,:,0]*(routscaLst[0][1]-routscaLst[0][0])
            # tempb = routscaLst[1][0] + rtwts[:,:,1]*(routscaLst[1][1]-routscaLst[1][0])
            tempa=routscaLst[0][0]+rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb=routscaLst[1][0]+rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])
            # routa = tempa.unsqueeze(-1)
            # routb = tempb.unsqueeze(-1)
            routa=tempa.repeat(Nstep,1).unsqueeze(-1)
            routb=tempb.repeat(Nstep,1).unsqueeze(-1)
            UH=UH_gamma(routa,routb,lenF=15)  # lenF: folter
            rf=torch.unsqueeze(Qsim,-1).permute([1,2,0])  # dim:gage*var*time
            UH=UH.permute([1,2,0])  # dim: gage*var*time
            Qsrout=UH_conv(rf,UH).permute([2,0,1])

            if comprout is True:  # Qs is [time, [gage*mult], var] now
                Qstemp=Qsrout.view(Nstep,Ngrid,mu)
                if muwts is None:
                    Qs=Qstemp.mean(-1,keepdim=True)
                else:
                    Qs=(Qstemp*muwts).sum(-1,keepdim=True)
            else:
                Qs=Qsrout

        else:  # no routing, output the initial average simulations
            Qs=torch.unsqueeze(Qsimave,-1)  # add a dimension

        if outstate is True:  # output states
            return Qs,SNOWPACK,MELTWATER,SM,SUZ,SLZ
        else:
            # return Qs
            Qall=torch.cat((Qs,Qsimave0,Qsimave1,Qsimave2,ETave),dim=-1)
            return Qall


class HBVMulTDET_1_1p(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""

    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMulTDET_1_1p,self).__init__()

    def forward(self,x,parameters,staind,tdlst,mu,muwts,rtwts,bufftime=0,outstate=False,routOpt=False,
                comprout=False,dydrop=False):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     X: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2]
        #     staind:use which time step from the learned para time series for static parameters
        #     tdlst: the index list of hbv parameters set as dynamic
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     dydrop: the possibility to drop a dynamic para to static to reduce potential overfitting
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET

        PRECS=1e-5  # keep the numerical calculation stable
        # bufftime=0 # pjli 20240926, donot understand why, comment it
        # Initialization to warm-up states
        if bufftime>0:
            with torch.no_grad():
                xinit=x[0:bufftime,:,:]
                initmodel=HBVMulTDET_1_1p()
                # buffpara=parameters[bufftime-1,:,:,:] # pjli 20240926 original
                buffpara = parameters[0:bufftime,:,:] # pql5336, dynamic parameter warm up
                Qsinit,SNOWPACK,MELTWATER,SM,SUZ,SLZ=initmodel(xinit,buffpara,staind,tdlst,mu,muwts,rtwts,
                                                               bufftime=0,outstate=True,routOpt=False,comprout=False,
                                                               dydrop=False)
        else:

            # Without warm-up bufftime=0, initialize state variables with zeros
            Ngrid=x.shape[1]
            SNOWPACK=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            MELTWATER=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            SM=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            SUZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            SLZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(x)

        P=x[bufftime:,:,0]
        Pm=P.unsqueeze(2).repeat(1,1,mu)  # precip
        T=x[bufftime:,:,1]
        Tm=T.unsqueeze(2).repeat(1,1,mu)  # temperature
        ETpot=x[bufftime:,:,2]
        ETpm=ETpot.unsqueeze(2).repeat(1,1,mu)  # potential ET
        parAll=parameters[bufftime:,:,:,:]
        parAllTrans=torch.zeros_like(parAll)

        ## scale the parameters to real values from [0,1]
        hbvscaLst=[[1,6],[50,1000],[0.05,0.9],[0.01,0.5],[0.001,0.2],[0.2,1],
                   [0,10],[0,100],[-2.5,2.5],[0.5,10],[0,0.1],[0,0.2],[0.3,5],[0,1]]  # HBV para
        routscaLst=[[0,2.9],[0,6.5]]  # routing para

        for ip in range(len(hbvscaLst)):  # not include routing. Scaling the parameters
            parAllTrans[:,:,ip,:]=hbvscaLst[ip][0]+parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        Nstep,Ngrid=P.size()

        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        parstaFull=parAllTrans[staind,:,:,:].unsqueeze(0).repeat([Nstep,1,1,1])  # static para matrix
        parhbvFull=torch.clone(parstaFull)
        # create probability mask for each parameter on the basin dimension
        pmat=torch.ones([1,Ngrid,1])*dydrop
        for ix in tdlst:
            staPar=parstaFull[:,:,ix-1,:]
            dynPar=parAllTrans[:,:,ix-1,:]
            drmask=torch.bernoulli(pmat).detach_().to(parhbvFull)  # to drop dynamic parameters as static in some basins
            comPar=dynPar*(1-drmask)+staPar*drmask
            parhbvFull[:,:,ix-1,:]=comPar

        # Initialize time series of model variables to save results
        Qsimmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        ETmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        SWEmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        # Output the box components of Q
        Qsimmu0=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu1=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu2=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)

        # # Not used. Logging the state variables for debug.
        # # SMlog = np.zeros(P.size())
        # logSM = np.zeros(P.size())
        # logPS = np.zeros(P.size())
        # logswet = np.zeros(P.size())
        # logRE = np.zeros(P.size())

        for t in range(Nstep):
            paraLst=[]
            for ip in range(len(hbvscaLst)):  # unpack HBV parameters
                paraLst.append(parhbvFull[t,:,ip,:])

            parBETA,parFC,parK0,parK1,parK2,parLP,parPERC,parUZL,parTT,parCFMAX,parCFR,parCWH,parBETAET,parC=paraLst
            # Separate precipitation into liquid and solid components
            PRECIP=Pm[t,:,:]
            TEMP=Tm[t,:,:]
            EVAP_P=ETpm[t,:,:]

            RAIN=torch.mul(PRECIP,(TEMP>=parTT).type(torch.float32))
            SNOW=torch.mul(PRECIP,(TEMP<parTT).type(torch.float32))

            # Snow process
            SNOWPACK=SNOWPACK+SNOW
            melt=parCFMAX*(TEMP-parTT)
            melt=torch.clamp(melt,min=0.0)
            melt=torch.min(melt,SNOWPACK)
            MELTWATER=MELTWATER+melt
            SNOWPACK=SNOWPACK-melt
            refreezing=parCFR*parCFMAX*(parTT-TEMP)
            refreezing=torch.clamp(refreezing,min=0.0)
            refreezing=torch.min(refreezing,MELTWATER)
            SNOWPACK=SNOWPACK+refreezing
            MELTWATER=MELTWATER-refreezing
            tosoil=MELTWATER-(parCWH*SNOWPACK)
            tosoil=torch.clamp(tosoil,min=0.0)
            MELTWATER=MELTWATER-tosoil

            # Soil and evaporation
            soil_wetness=(SM/parFC)**parBETA
            soil_wetness=torch.clamp(soil_wetness,min=0.0,max=1.0)
            recharge=(RAIN+tosoil)*soil_wetness

            # Not used, logging states for checking
            # logSM[t,:] = SM.detach().cpu().numpy()
            # logPS[t,:] = (RAIN + tosoil).detach().cpu().numpy()
            # logswet[t,:] = (SM / parFC).detach().cpu().numpy()
            # logRE[t, :] = recharge.detach().cpu().numpy()

            SM=SM+RAIN+tosoil-recharge
            excess=SM-parFC
            excess=torch.clamp(excess,min=0.0)
            SM=SM-excess
            # MODIFY here. Different as class HBVMulT(). Add a ET shape parameter parBETAET
            evapfactor=(SM/(parLP*parFC))**parBETAET
            evapfactor=torch.clamp(evapfactor,min=0.0,max=1.0)
            ETact=EVAP_P*evapfactor
            ETact=torch.min(SM,ETact)
            SM=torch.clamp(SM-ETact,min=PRECS)  # SM can not be zero for gradient tracking

            capillary=torch.min(SLZ,parC*SLZ*(1.0-torch.clamp(SM/parFC,max=1.0)))

            SM=torch.clamp(SM+capillary,min=PRECS)
            SLZ=torch.clamp(SLZ-capillary,min=PRECS)

            # Groundwater boxes
            SUZ=SUZ+recharge+excess
            PERC=torch.min(SUZ,parPERC)
            SUZ=SUZ-PERC
            Q0=parK0*torch.clamp(SUZ-parUZL,min=0.0)
            SUZ=SUZ-Q0
            Q1=parK1*SUZ
            SUZ=SUZ-Q1
            SLZ=SLZ+PERC
            Q2=parK2*SLZ
            SLZ=SLZ-Q2
            Qsimmu[t,:,:]=Q0+Q1+Q2

            # save components for Q
            Qsimmu0[t,:,:]=Q0
            Qsimmu1[t,:,:]=Q1
            Qsimmu2[t,:,:]=Q2
            ETmu[t,:,:]=ETact
            SWEmu[t,:,:]=SNOWPACK
            # Not used, for debug state variables
            # SMlog[t,:, :] = SM.detach().cpu().numpy()
            # SUZlog[t,:,:] = SUZ.detach().cpu().numpy()
            # SLZlog[t,:,:] = SLZ.detach().cpu().numpy()

        Qsimave0=Qsimmu0.mean(-1,keepdim=True)
        Qsimave1=Qsimmu1.mean(-1,keepdim=True)
        Qsimave2=Qsimmu2.mean(-1,keepdim=True)
        ETave=ETmu.mean(-1,keepdim=True)
        SWEave=SWEmu.mean(-1,keepdim=True)

        # get the initial average
        if muwts is None:  # simple average
            Qsimave=Qsimmu.mean(-1)
        else:  # weighted average using learned weights
            Qsimave=(Qsimmu*muwts).sum(-1)

        if routOpt is True:  # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim=Qsimmu.view(Nstep,Ngrid*mu)
            else:
                # average the components, then do routing
                Qsim=Qsimave

            # scale two routing parameters
            tempa=routscaLst[0][0]+rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb=routscaLst[1][0]+rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])
            routa=tempa.repeat(Nstep,1).unsqueeze(-1)
            routb=tempb.repeat(Nstep,1).unsqueeze(-1)
            UH=UH_gamma(routa,routb,lenF=15)  # lenF: folter
            rf=torch.unsqueeze(Qsim,-1).permute([1,2,0])  # dim:gage*var*time
            UH=UH.permute([1,2,0])  # dim: gage*var*time
            Qsrout=UH_conv(rf,UH).permute([2,0,1])

            if comprout is True:  # Qs is [time, [gage*mult], var] now
                Qstemp=Qsrout.view(Nstep,Ngrid,mu)
                if muwts is None:
                    Qs=Qstemp.mean(-1,keepdim=True)
                else:
                    Qs=(Qstemp*muwts).sum(-1,keepdim=True)
            else:
                Qs=Qsrout

        else:  # no routing, output the initial average simulations
            Qs=torch.unsqueeze(Qsimave,-1)  # add a dimension

        if outstate is True:  # output states
            return Qs,SNOWPACK,MELTWATER,SM,SUZ,SLZ
        else:
            # return Qs
            Qall=torch.cat((Qs,Qsimave0,Qsimave1,Qsimave2,ETave,SWEave),dim=-1)
            return Qall


class HBVMulTDET_1_1p_no_g(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""

    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMulTDET_1_1p_no_g,self).__init__()

    def forward(self,x,parameters,staind,tdlst,mu,muwts,rtwts,bufftime=0,outstate=False,routOpt=False,
                comprout=False,dydrop=False):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     X: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2]
        #     staind:use whissssssssch time step from the learned para time series for static parameters
        #     tdlst: the index list of hbv parameters set as dynamic
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     dydrop: the possibility to drop a dynamic para to static to reduce potential overfitting
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET

        PRECS=1e-5  # keep the numerical calculation stable
        bufftime_old = bufftime
        bufftime=0 # pjli 20240926, donot understand why, comment it
        # # Initialization to warm-up states
        # if bufftime>0:
        #     with torch.no_grad():
        #         xinit=x[0:bufftime,:,:]
        #         initmodel=HBVMulTDET_1_1p()
        #         # buffpara=parameters[bufftime-1,:,:,:] # pjli 20240926 original
        #         buffpara = parameters[0:bufftime,:,:] # pql5336, dynamic parameter warm up
        #         Qsinit,SNOWPACK,MELTWATER,SM,SUZ,SLZ=initmodel(xinit,buffpara,staind,tdlst,mu,muwts,rtwts,
        #                                                        bufftime=0,outstate=True,routOpt=False,comprout=False,
        #                                                        dydrop=False)
        # else:

        # Without warm-up bufftime=0, initialize state variables with zeros
        Ngrid=x.shape[1]
        SNOWPACK=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        MELTWATER=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SM=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SUZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SLZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(x)

        P=x[bufftime:,:,0]
        Pm=P.unsqueeze(2).repeat(1,1,mu)  # precip
        T=x[bufftime:,:,1]
        Tm=T.unsqueeze(2).repeat(1,1,mu)  # temperature
        ETpot=x[bufftime:,:,2]
        ETpm=ETpot.unsqueeze(2).repeat(1,1,mu)  # potential ET
        parAll=parameters[bufftime:,:,:,:]
        parAllTrans=torch.zeros_like(parAll)

        ## scale the parameters to real values from [0,1]
        hbvscaLst=[[1,6],[50,1000],[0.05,0.9],[0.01,0.5],[0.001,0.2],[0.2,1],
                   [0,10],[0,100],[-2.5,2.5],[0.5,10],[0,0.1],[0,0.2],[0.3,5],[0,1]]  # HBV para
        routscaLst=[[0,2.9],[0,6.5]]  # routing para

        for ip in range(len(hbvscaLst)):  # not include routing. Scaling the parameters
            parAllTrans[:,:,ip,:]=hbvscaLst[ip][0]+parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        Nstep,Ngrid=P.size()

        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        parstaFull=parAllTrans[staind,:,:,:].unsqueeze(0).repeat([Nstep,1,1,1])  # static para matrix
        parhbvFull=torch.clone(parstaFull)
        # create probability mask for each parameter on the basin dimension
        pmat=torch.ones([1,Ngrid,1])*dydrop
        for ix in tdlst:
            staPar=parstaFull[:,:,ix-1,:]
            dynPar=parAllTrans[:,:,ix-1,:]
            drmask=torch.bernoulli(pmat).detach_().to(parhbvFull)  # to drop dynamic parameters as static in some basins
            comPar=dynPar*(1-drmask)+staPar*drmask
            parhbvFull[:,:,ix-1,:]=comPar

        # Initialize time series of model variables to save results
        Qsimmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        ETmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        SWEmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        # Output the box components of Q
        Qsimmu0=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu1=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu2=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)

        # # Not used. Logging the state variables for debug.
        # # SMlog = np.zeros(P.size())
        # logSM = np.zeros(P.size())
        # logPS = np.zeros(P.size())
        # logswet = np.zeros(P.size())
        # logRE = np.zeros(P.size())

        for t in range(Nstep):
            with torch.no_grad() if t<bufftime_old else torch.enable_grad():
                paraLst=[]
                for ip in range(len(hbvscaLst)):  # unpack HBV parameters
                    paraLst.append(parhbvFull[t,:,ip,:])
    
                parBETA,parFC,parK0,parK1,parK2,parLP,parPERC,parUZL,parTT,parCFMAX,parCFR,parCWH,parBETAET,parC=paraLst
                # Separate precipitation into liquid and solid components
                PRECIP=Pm[t,:,:]
                TEMP=Tm[t,:,:]
                EVAP_P=ETpm[t,:,:]
    
                RAIN=torch.mul(PRECIP,(TEMP>=parTT).type(torch.float32))
                SNOW=torch.mul(PRECIP,(TEMP<parTT).type(torch.float32))
    
                # Snow process
                SNOWPACK=SNOWPACK+SNOW
                melt=parCFMAX*(TEMP-parTT)
                melt=torch.clamp(melt,min=0.0)
                melt=torch.min(melt,SNOWPACK)
                MELTWATER=MELTWATER+melt
                SNOWPACK=SNOWPACK-melt
                refreezing=parCFR*parCFMAX*(parTT-TEMP)
                refreezing=torch.clamp(refreezing,min=0.0)
                refreezing=torch.min(refreezing,MELTWATER)
                SNOWPACK=SNOWPACK+refreezing
                MELTWATER=MELTWATER-refreezing
                tosoil=MELTWATER-(parCWH*SNOWPACK)
                tosoil=torch.clamp(tosoil,min=0.0)
                MELTWATER=MELTWATER-tosoil
    
                # Soil and evaporation
                soil_wetness=(SM/parFC)**parBETA
                soil_wetness=torch.clamp(soil_wetness,min=0.0,max=1.0)
                recharge=(RAIN+tosoil)*soil_wetness
    
                # Not used, logging states for checking
                # logSM[t,:] = SM.detach().cpu().numpy()
                # logPS[t,:] = (RAIN + tosoil).detach().cpu().numpy()
                # logswet[t,:] = (SM / parFC).detach().cpu().numpy()
                # logRE[t, :] = recharge.detach().cpu().numpy()
    
                SM=SM+RAIN+tosoil-recharge
                excess=SM-parFC
                excess=torch.clamp(excess,min=0.0)
                SM=SM-excess
                # MODIFY here. Different as class HBVMulT(). Add a ET shape parameter parBETAET
                evapfactor=(SM/(parLP*parFC))**parBETAET
                evapfactor=torch.clamp(evapfactor,min=0.0,max=1.0)
                ETact=EVAP_P*evapfactor
                ETact=torch.min(SM,ETact)
                SM=torch.clamp(SM-ETact,min=PRECS)  # SM can not be zero for gradient tracking
    
                capillary=torch.min(SLZ,parC*SLZ*(1.0-torch.clamp(SM/parFC,max=1.0)))
    
                SM=torch.clamp(SM+capillary,min=PRECS)
                SLZ=torch.clamp(SLZ-capillary,min=PRECS)
    
                # Groundwater boxes
                SUZ=SUZ+recharge+excess
                PERC=torch.min(SUZ,parPERC)
                SUZ=SUZ-PERC
                Q0=parK0*torch.clamp(SUZ-parUZL,min=0.0)
                SUZ=SUZ-Q0
                Q1=parK1*SUZ
                SUZ=SUZ-Q1
                SLZ=SLZ+PERC
                Q2=parK2*SLZ
                SLZ=SLZ-Q2
                Qsimmu[t,:,:]=Q0+Q1+Q2
    
                # save components for Q
                Qsimmu0[t,:,:]=Q0
                Qsimmu1[t,:,:]=Q1
                Qsimmu2[t,:,:]=Q2
                ETmu[t,:,:]=ETact
                SWEmu[t,:,:]=SNOWPACK
                # Not used, for debug state variables
                # SMlog[t,:, :] = SM.detach().cpu().numpy()
                # SUZlog[t,:,:] = SUZ.detach().cpu().numpy()
                # SLZlog[t,:,:] = SLZ.detach().cpu().numpy()

        Qsimave0=Qsimmu0.mean(-1,keepdim=True)
        Qsimave1=Qsimmu1.mean(-1,keepdim=True)
        Qsimave2=Qsimmu2.mean(-1,keepdim=True)
        ETave=ETmu.mean(-1,keepdim=True)
        SWEave=SWEmu.mean(-1,keepdim=True)

        # get the initial average
        if muwts is None:  # simple average
            Qsimave=Qsimmu.mean(-1)
        else:  # weighted average using learned weights
            Qsimave=(Qsimmu*muwts).sum(-1)

        if routOpt is True:  # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim=Qsimmu.view(Nstep,Ngrid*mu)
            else:
                # average the components, then do routing
                Qsim=Qsimave

            # scale two routing parameters
            tempa=routscaLst[0][0]+rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb=routscaLst[1][0]+rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])
            routa=tempa.repeat(Nstep,1).unsqueeze(-1)
            routb=tempb.repeat(Nstep,1).unsqueeze(-1)
            UH=UH_gamma(routa,routb,lenF=15)  # lenF: folter
            rf=torch.unsqueeze(Qsim,-1).permute([1,2,0])  # dim:gage*var*time
            UH=UH.permute([1,2,0])  # dim: gage*var*time
            Qsrout=UH_conv(rf,UH).permute([2,0,1])

            if comprout is True:  # Qs is [time, [gage*mult], var] now
                Qstemp=Qsrout.view(Nstep,Ngrid,mu)
                if muwts is None:
                    Qs=Qstemp.mean(-1,keepdim=True)
                else:
                    Qs=(Qstemp*muwts).sum(-1,keepdim=True)
            else:
                Qs=Qsrout

        else:  # no routing, output the initial average simulations
            Qs=torch.unsqueeze(Qsimave,-1)  # add a dimension

        if outstate is True:  # output states
            return Qs,SNOWPACK,MELTWATER,SM,SUZ,SLZ
        else:
            # return Qs
            Qall=torch.cat((Qs,Qsimave0,Qsimave1,Qsimave2,ETave,SWEave),dim=-1)
            return Qall[bufftime_old:]


class HBVMulTDET_1_1p_no_g_05(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""

    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMulTDET_1_1p_no_g_05,self).__init__()

    def forward(self,x,parameters,staind,tdlst,mu,muwts,rtwts,bufftime=0,outstate=False,routOpt=False,
                comprout=False,dydrop=False):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     X: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2]
        #     staind:use whissssssssch time step from the learned para time series for static parameters
        #     tdlst: the index list of hbv parameters set as dynamic
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     dydrop: the possibility to drop a dynamic para to static to reduce potential overfitting
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET

        PRECS=1e-5  # keep the numerical calculation stable
        bufftime_old=bufftime
        bufftime=0  # pjli 20240926, donot understand why, comment it
        # # Initialization to warm-up states
        # if bufftime>0:
        #     with torch.no_grad():
        #         xinit=x[0:bufftime,:,:]
        #         initmodel=HBVMulTDET_1_1p()
        #         # buffpara=parameters[bufftime-1,:,:,:] # pjli 20240926 original
        #         buffpara = parameters[0:bufftime,:,:] # pql5336, dynamic parameter warm up
        #         Qsinit,SNOWPACK,MELTWATER,SM,SUZ,SLZ=initmodel(xinit,buffpara,staind,tdlst,mu,muwts,rtwts,
        #                                                        bufftime=0,outstate=True,routOpt=False,comprout=False,
        #                                                        dydrop=False)
        # else:

        # Without warm-up bufftime=0, initialize state variables with zeros
        Ngrid=x.shape[1]
        SNOWPACK=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        MELTWATER=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SM=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SUZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SLZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(x)

        P=x[bufftime:,:,0]
        Pm=P.unsqueeze(2).repeat(1,1,mu)  # precip
        T=x[bufftime:,:,1]
        Tm=T.unsqueeze(2).repeat(1,1,mu)  # temperature
        ETpot=x[bufftime:,:,2]
        ETpm=ETpot.unsqueeze(2).repeat(1,1,mu)  # potential ET
        parAll=parameters[bufftime:,:,:,:]
        parAllTrans=torch.zeros_like(parAll)

        ## scale the parameters to real values from [0,1]
        hbvscaLst=[[0.5,6],[50,1000],[0.05,0.9],[0.01,0.5],[0.001,0.2],[0.2,1],
                   [0,10],[0,100],[-2.5,2.5],[0.5,10],[0,0.1],[0,0.2],[0.3,5],[0,1]]  # HBV para
        routscaLst=[[0,2.9],[0,6.5]]  # routing para

        for ip in range(len(hbvscaLst)):  # not include routing. Scaling the parameters
            parAllTrans[:,:,ip,:]=hbvscaLst[ip][0]+parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        Nstep,Ngrid=P.size()

        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        parstaFull=parAllTrans[staind,:,:,:].unsqueeze(0).repeat([Nstep,1,1,1])  # static para matrix
        parhbvFull=torch.clone(parstaFull)
        # create probability mask for each parameter on the basin dimension
        pmat=torch.ones([1,Ngrid,1])*dydrop
        for ix in tdlst:
            staPar=parstaFull[:,:,ix-1,:]
            dynPar=parAllTrans[:,:,ix-1,:]
            drmask=torch.bernoulli(pmat).detach_().to(parhbvFull)  # to drop dynamic parameters as static in some basins
            comPar=dynPar*(1-drmask)+staPar*drmask
            parhbvFull[:,:,ix-1,:]=comPar

        # Initialize time series of model variables to save results
        Qsimmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        ETmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        SWEmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        # Output the box components of Q
        Qsimmu0=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu1=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu2=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)

        # # Not used. Logging the state variables for debug.
        # # SMlog = np.zeros(P.size())
        # logSM = np.zeros(P.size())
        # logPS = np.zeros(P.size())
        # logswet = np.zeros(P.size())
        # logRE = np.zeros(P.size())

        for t in range(Nstep):
            with torch.no_grad() if t<bufftime_old else torch.enable_grad():
                paraLst=[]
                for ip in range(len(hbvscaLst)):  # unpack HBV parameters
                    paraLst.append(parhbvFull[t,:,ip,:])

                parBETA,parFC,parK0,parK1,parK2,parLP,parPERC,parUZL,parTT,parCFMAX,parCFR,parCWH,parBETAET,parC=paraLst
                # Separate precipitation into liquid and solid components
                PRECIP=Pm[t,:,:]
                TEMP=Tm[t,:,:]
                EVAP_P=ETpm[t,:,:]

                RAIN=torch.mul(PRECIP,(TEMP>=parTT).type(torch.float32))
                SNOW=torch.mul(PRECIP,(TEMP<parTT).type(torch.float32))

                # Snow process
                SNOWPACK=SNOWPACK+SNOW
                melt=parCFMAX*(TEMP-parTT)
                melt=torch.clamp(melt,min=0.0)
                melt=torch.min(melt,SNOWPACK)
                MELTWATER=MELTWATER+melt
                SNOWPACK=SNOWPACK-melt
                refreezing=parCFR*parCFMAX*(parTT-TEMP)
                refreezing=torch.clamp(refreezing,min=0.0)
                refreezing=torch.min(refreezing,MELTWATER)
                SNOWPACK=SNOWPACK+refreezing
                MELTWATER=MELTWATER-refreezing
                tosoil=MELTWATER-(parCWH*SNOWPACK)
                tosoil=torch.clamp(tosoil,min=0.0)
                MELTWATER=MELTWATER-tosoil

                # Soil and evaporation
                soil_wetness=(SM/parFC)**parBETA
                soil_wetness=torch.clamp(soil_wetness,min=0.0,max=1.0)
                recharge=(RAIN+tosoil)*soil_wetness

                # Not used, logging states for checking
                # logSM[t,:] = SM.detach().cpu().numpy()
                # logPS[t,:] = (RAIN + tosoil).detach().cpu().numpy()
                # logswet[t,:] = (SM / parFC).detach().cpu().numpy()
                # logRE[t, :] = recharge.detach().cpu().numpy()

                SM=SM+RAIN+tosoil-recharge
                excess=SM-parFC
                excess=torch.clamp(excess,min=0.0)
                SM=SM-excess
                # MODIFY here. Different as class HBVMulT(). Add a ET shape parameter parBETAET
                evapfactor=(SM/(parLP*parFC))**parBETAET
                evapfactor=torch.clamp(evapfactor,min=0.0,max=1.0)
                ETact=EVAP_P*evapfactor
                ETact=torch.min(SM,ETact)
                SM=torch.clamp(SM-ETact,min=PRECS)  # SM can not be zero for gradient tracking

                capillary=torch.min(SLZ,parC*SLZ*(1.0-torch.clamp(SM/parFC,max=1.0)))

                SM=torch.clamp(SM+capillary,min=PRECS)
                SLZ=torch.clamp(SLZ-capillary,min=PRECS)

                # Groundwater boxes
                SUZ=SUZ+recharge+excess
                PERC=torch.min(SUZ,parPERC)
                SUZ=SUZ-PERC
                Q0=parK0*torch.clamp(SUZ-parUZL,min=0.0)
                SUZ=SUZ-Q0
                Q1=parK1*SUZ
                SUZ=SUZ-Q1
                SLZ=SLZ+PERC
                Q2=parK2*SLZ
                SLZ=SLZ-Q2
                Qsimmu[t,:,:]=Q0+Q1+Q2

                # save components for Q
                Qsimmu0[t,:,:]=Q0
                Qsimmu1[t,:,:]=Q1
                Qsimmu2[t,:,:]=Q2
                ETmu[t,:,:]=ETact
                SWEmu[t,:,:]=SNOWPACK
                # Not used, for debug state variables
                # SMlog[t,:, :] = SM.detach().cpu().numpy()
                # SUZlog[t,:,:] = SUZ.detach().cpu().numpy()
                # SLZlog[t,:,:] = SLZ.detach().cpu().numpy()

        Qsimave0=Qsimmu0.mean(-1,keepdim=True)
        Qsimave1=Qsimmu1.mean(-1,keepdim=True)
        Qsimave2=Qsimmu2.mean(-1,keepdim=True)
        ETave=ETmu.mean(-1,keepdim=True)
        SWEave=SWEmu.mean(-1,keepdim=True)

        # get the initial average
        if muwts is None:  # simple average
            Qsimave=Qsimmu.mean(-1)
        else:  # weighted average using learned weights
            Qsimave=(Qsimmu*muwts).sum(-1)

        if routOpt is True:  # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim=Qsimmu.view(Nstep,Ngrid*mu)
            else:
                # average the components, then do routing
                Qsim=Qsimave

            # scale two routing parameters
            tempa=routscaLst[0][0]+rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb=routscaLst[1][0]+rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])
            routa=tempa.repeat(Nstep,1).unsqueeze(-1)
            routb=tempb.repeat(Nstep,1).unsqueeze(-1)
            UH=UH_gamma(routa,routb,lenF=15)  # lenF: folter
            rf=torch.unsqueeze(Qsim,-1).permute([1,2,0])  # dim:gage*var*time
            UH=UH.permute([1,2,0])  # dim: gage*var*time
            Qsrout=UH_conv(rf,UH).permute([2,0,1])

            if comprout is True:  # Qs is [time, [gage*mult], var] now
                Qstemp=Qsrout.view(Nstep,Ngrid,mu)
                if muwts is None:
                    Qs=Qstemp.mean(-1,keepdim=True)
                else:
                    Qs=(Qstemp*muwts).sum(-1,keepdim=True)
            else:
                Qs=Qsrout

        else:  # no routing, output the initial average simulations
            Qs=torch.unsqueeze(Qsimave,-1)  # add a dimension

        if outstate is True:  # output states
            return Qs,SNOWPACK,MELTWATER,SM,SUZ,SLZ
        else:
            # return Qs
            Qall=torch.cat((Qs,Qsimave0,Qsimave1,Qsimave2,ETave,SWEave),dim=-1)
            return Qall[bufftime_old:]


class HBVMulTDET_1_1p_PE_bias(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""

    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMulTDET_1_1p_PE_bias,self).__init__()

    def forward(self,x,parameters,staind,tdlst,mu,muwts,rtwts,bufftime=0,outstate=False,routOpt=False,
                comprout=False,dydrop=False):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     X: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2]
        #     staind:use which time step from the learned para time series for static parameters
        #     tdlst: the index list of hbv parameters set as dynamic
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     dydrop: the possibility to drop a dynamic para to static to reduce potential overfitting
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET

        PRECS=1e-5  # keep the numerical calculation stable
        # bufftime=0 # pjli 20240926, donot understand why, comment it
        # Initialization to warm-up states
        if bufftime>0:
            with torch.no_grad():
                xinit=x[0:bufftime,:,:]
                initmodel=HBVMulTDET_1_1p()
                # buffpara=parameters[bufftime-1,:,:,:] # pjli 20240926 original
                buffpara = parameters[0:bufftime,:,:] # pql5336, dynamic parameter warm up
                Qsinit,SNOWPACK,MELTWATER,SM,SUZ,SLZ=initmodel(xinit,buffpara,staind,tdlst,mu,muwts,rtwts,
                                                               bufftime=0,outstate=True,routOpt=False,comprout=False,
                                                               dydrop=False)
        else:

            # Without warm-up bufftime=0, initialize state variables with zeros
            Ngrid=x.shape[1]
            SNOWPACK=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            MELTWATER=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            SM=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            SUZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            SLZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(x)

        P=x[bufftime:,:,0]
        Pm=P.unsqueeze(2).repeat(1,1,mu)  # precip
        T=x[bufftime:,:,1]
        Tm=T.unsqueeze(2).repeat(1,1,mu)  # temperature
        ETpot=x[bufftime:,:,2]
        ETpm=ETpot.unsqueeze(2).repeat(1,1,mu)  # potential ET
        parAll=parameters[bufftime:,:,:,:]
        parAllTrans=torch.zeros_like(parAll)

        ## scale the parameters to real values from [0,1]
        hbvscaLst=[[1,6],[50,1000],[0.05,0.9],[0.01,0.5],[0.001,0.2],[0.2,1],
                   [0,10],[0,100],[-2.5,2.5],[0.5,10],[0,0.1],[0,0.2],[0.3,5],[0,1],[-0.1,0.1]]  # HBV para
        routscaLst=[[0,2.9],[0,6.5]]  # routing para

        for ip in range(len(hbvscaLst)):  # not include routing. Scaling the parameters
            parAllTrans[:,:,ip,:]=hbvscaLst[ip][0]+parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        Nstep,Ngrid=P.size()

        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        parstaFull=parAllTrans[staind,:,:,:].unsqueeze(0).repeat([Nstep,1,1,1])  # static para matrix
        parhbvFull=torch.clone(parstaFull)
        # create probability mask for each parameter on the basin dimension
        pmat=torch.ones([1,Ngrid,1])*dydrop
        for ix in tdlst:
            staPar=parstaFull[:,:,ix-1,:]
            dynPar=parAllTrans[:,:,ix-1,:]
            drmask=torch.bernoulli(pmat).detach_().to(parhbvFull)  # to drop dynamic parameters as static in some basins
            comPar=dynPar*(1-drmask)+staPar*drmask
            parhbvFull[:,:,ix-1,:]=comPar

        # Initialize time series of model variables to save results
        Qsimmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        ETmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        SWEmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        # Output the box components of Q
        Qsimmu0=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu1=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu2=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)

        # # Not used. Logging the state variables for debug.
        # # SMlog = np.zeros(P.size())
        # logSM = np.zeros(P.size())
        # logPS = np.zeros(P.size())
        # logswet = np.zeros(P.size())
        # logRE = np.zeros(P.size())

        for t in range(Nstep):
            paraLst=[]
            for ip in range(len(hbvscaLst)):  # unpack HBV parameters
                paraLst.append(parhbvFull[t,:,ip,:])

            parBETA,parFC,parK0,parK1,parK2,parLP,parPERC,parUZL,parTT,parCFMAX,parCFR,parCWH,parBETAET,parC, P_bias=paraLst
            # Separate precipitation into liquid and solid components
            PRECIP=Pm[t,:,:]+P_bias*Pm[t,:,:]
            TEMP=Tm[t,:,:]
            EVAP_P=ETpm[t,:,:]

            RAIN=torch.mul(PRECIP,(TEMP>=parTT).type(torch.float32))
            SNOW=torch.mul(PRECIP,(TEMP<parTT).type(torch.float32))

            # Snow process
            SNOWPACK=SNOWPACK+SNOW
            melt=parCFMAX*(TEMP-parTT)
            melt=torch.clamp(melt,min=0.0)
            melt=torch.min(melt,SNOWPACK)
            MELTWATER=MELTWATER+melt
            SNOWPACK=SNOWPACK-melt
            refreezing=parCFR*parCFMAX*(parTT-TEMP)
            refreezing=torch.clamp(refreezing,min=0.0)
            refreezing=torch.min(refreezing,MELTWATER)
            SNOWPACK=SNOWPACK+refreezing
            MELTWATER=MELTWATER-refreezing
            tosoil=MELTWATER-(parCWH*SNOWPACK)
            tosoil=torch.clamp(tosoil,min=0.0)
            MELTWATER=MELTWATER-tosoil

            # Soil and evaporation
            soil_wetness=(SM/parFC)**parBETA
            soil_wetness=torch.clamp(soil_wetness,min=0.0,max=1.0)
            recharge=(RAIN+tosoil)*soil_wetness

            # Not used, logging states for checking
            # logSM[t,:] = SM.detach().cpu().numpy()
            # logPS[t,:] = (RAIN + tosoil).detach().cpu().numpy()
            # logswet[t,:] = (SM / parFC).detach().cpu().numpy()
            # logRE[t, :] = recharge.detach().cpu().numpy()

            SM=SM+RAIN+tosoil-recharge
            excess=SM-parFC
            excess=torch.clamp(excess,min=0.0)
            SM=SM-excess
            # MODIFY here. Different as class HBVMulT(). Add a ET shape parameter parBETAET
            evapfactor=(SM/(parLP*parFC))**parBETAET
            evapfactor=torch.clamp(evapfactor,min=0.0,max=1.0)
            ETact=EVAP_P*evapfactor
            ETact=torch.min(SM,ETact)
            SM=torch.clamp(SM-ETact,min=PRECS)  # SM can not be zero for gradient tracking

            capillary=torch.min(SLZ,parC*SLZ*(1.0-torch.clamp(SM/parFC,max=1.0)))

            SM=torch.clamp(SM+capillary,min=PRECS)
            SLZ=torch.clamp(SLZ-capillary,min=PRECS)

            # Groundwater boxes
            SUZ=SUZ+recharge+excess
            PERC=torch.min(SUZ,parPERC)
            SUZ=SUZ-PERC
            Q0=parK0*torch.clamp(SUZ-parUZL,min=0.0)
            SUZ=SUZ-Q0
            Q1=parK1*SUZ
            SUZ=SUZ-Q1
            SLZ=SLZ+PERC
            Q2=parK2*SLZ
            SLZ=SLZ-Q2
            Qsimmu[t,:,:]=Q0+Q1+Q2

            # save components for Q
            Qsimmu0[t,:,:]=Q0
            Qsimmu1[t,:,:]=Q1
            Qsimmu2[t,:,:]=Q2
            ETmu[t,:,:]=ETact
            SWEmu[t,:,:]=SNOWPACK
            # Not used, for debug state variables
            # SMlog[t,:, :] = SM.detach().cpu().numpy()
            # SUZlog[t,:,:] = SUZ.detach().cpu().numpy()
            # SLZlog[t,:,:] = SLZ.detach().cpu().numpy()

        Qsimave0=Qsimmu0.mean(-1,keepdim=True)
        Qsimave1=Qsimmu1.mean(-1,keepdim=True)
        Qsimave2=Qsimmu2.mean(-1,keepdim=True)
        ETave=ETmu.mean(-1,keepdim=True)
        SWEave=SWEmu.mean(-1,keepdim=True)

        # get the initial average
        if muwts is None:  # simple average
            Qsimave=Qsimmu.mean(-1)
        else:  # weighted average using learned weights
            Qsimave=(Qsimmu*muwts).sum(-1)

        if routOpt is True:  # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim=Qsimmu.view(Nstep,Ngrid*mu)
            else:
                # average the components, then do routing
                Qsim=Qsimave

            # scale two routing parameters
            tempa=routscaLst[0][0]+rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb=routscaLst[1][0]+rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])
            routa=tempa.repeat(Nstep,1).unsqueeze(-1)
            routb=tempb.repeat(Nstep,1).unsqueeze(-1)
            UH=UH_gamma(routa,routb,lenF=15)  # lenF: folter
            rf=torch.unsqueeze(Qsim,-1).permute([1,2,0])  # dim:gage*var*time
            UH=UH.permute([1,2,0])  # dim: gage*var*time
            Qsrout=UH_conv(rf,UH).permute([2,0,1])

            if comprout is True:  # Qs is [time, [gage*mult], var] now
                Qstemp=Qsrout.view(Nstep,Ngrid,mu)
                if muwts is None:
                    Qs=Qstemp.mean(-1,keepdim=True)
                else:
                    Qs=(Qstemp*muwts).sum(-1,keepdim=True)
            else:
                Qs=Qsrout

        else:  # no routing, output the initial average simulations
            Qs=torch.unsqueeze(Qsimave,-1)  # add a dimension

        if outstate is True:  # output states
            return Qs,SNOWPACK,MELTWATER,SM,SUZ,SLZ
        else:
            # return Qs
            Qall=torch.cat((Qs,Qsimave0,Qsimave1,Qsimave2,ETave,SWEave),dim=-1)
            return Qall

class HBVMulTDET_1_1p_PE_bias2(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""

    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first
    # not only P bais but also PET bias

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMulTDET_1_1p_PE_bias2,self).__init__()

    def forward(self,x,parameters,staind,tdlst,mu,muwts,rtwts,bufftime=0,outstate=False,routOpt=False,
                comprout=False,dydrop=False):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     X: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2]
        #     staind:use which time step from the learned para time series for static parameters
        #     tdlst: the index list of hbv parameters set as dynamic
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     dydrop: the possibility to drop a dynamic para to static to reduce potential overfitting
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET

        PRECS=1e-5  # keep the numerical calculation stable
        # bufftime=0 # pjli 20240926, donot understand why, comment it
        # Initialization to warm-up states
        if bufftime>0:
            with torch.no_grad():
                xinit=x[0:bufftime,:,:]
                initmodel=HBVMulTDET_1_1p()
                # buffpara=parameters[bufftime-1,:,:,:] # pjli 20240926 original
                buffpara = parameters[0:bufftime,:,:] # pql5336, dynamic parameter warm up
                Qsinit,SNOWPACK,MELTWATER,SM,SUZ,SLZ=initmodel(xinit,buffpara,staind,tdlst,mu,muwts,rtwts,
                                                               bufftime=0,outstate=True,routOpt=False,comprout=False,
                                                               dydrop=False)
        else:

            # Without warm-up bufftime=0, initialize state variables with zeros
            Ngrid=x.shape[1]
            SNOWPACK=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            MELTWATER=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            SM=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            SUZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            SLZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(x)

        P=x[bufftime:,:,0]
        Pm=P.unsqueeze(2).repeat(1,1,mu)  # precip
        T=x[bufftime:,:,1]
        Tm=T.unsqueeze(2).repeat(1,1,mu)  # temperature
        ETpot=x[bufftime:,:,2]
        ETpm=ETpot.unsqueeze(2).repeat(1,1,mu)  # potential ET
        parAll=parameters[bufftime:,:,:,:]
        parAllTrans=torch.zeros_like(parAll)

        ## scale the parameters to real values from [0,1]
        hbvscaLst=[[1,6],[50,1000],[0.05,0.9],[0.01,0.5],[0.001,0.2],[0.2,1],
                   [0,10],[0,100],[-2.5,2.5],[0.5,10],[0,0.1],[0,0.2],[0.3,5],[0,1],[-0.1,0.1],[-0.1,0.1]]  # HBV para
        routscaLst=[[0,2.9],[0,6.5]]  # routing para

        for ip in range(len(hbvscaLst)):  # not include routing. Scaling the parameters
            parAllTrans[:,:,ip,:]=hbvscaLst[ip][0]+parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        Nstep,Ngrid=P.size()

        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        parstaFull=parAllTrans[staind,:,:,:].unsqueeze(0).repeat([Nstep,1,1,1])  # static para matrix
        parhbvFull=torch.clone(parstaFull)
        # create probability mask for each parameter on the basin dimension
        pmat=torch.ones([1,Ngrid,1])*dydrop
        for ix in tdlst:
            staPar=parstaFull[:,:,ix-1,:]
            dynPar=parAllTrans[:,:,ix-1,:]
            drmask=torch.bernoulli(pmat).detach_().to(parhbvFull)  # to drop dynamic parameters as static in some basins
            comPar=dynPar*(1-drmask)+staPar*drmask
            parhbvFull[:,:,ix-1,:]=comPar

        # Initialize time series of model variables to save results
        Qsimmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        ETmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        SWEmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        # Output the box components of Q
        Qsimmu0=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu1=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu2=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)

        # # Not used. Logging the state variables for debug.
        # # SMlog = np.zeros(P.size())
        # logSM = np.zeros(P.size())
        # logPS = np.zeros(P.size())
        # logswet = np.zeros(P.size())
        # logRE = np.zeros(P.size())

        for t in range(Nstep):
            paraLst=[]
            for ip in range(len(hbvscaLst)):  # unpack HBV parameters
                paraLst.append(parhbvFull[t,:,ip,:])

            parBETA,parFC,parK0,parK1,parK2,parLP,parPERC,parUZL,parTT,parCFMAX,parCFR,parCWH,parBETAET,parC, P_bias, E_bias=paraLst
            # Separate precipitation into liquid and solid components
            PRECIP=Pm[t,:,:]+P_bias*Pm[t,:,:]
            TEMP=Tm[t,:,:]
            EVAP_P=ETpm[t,:,:]+E_bias*ETpm[t,:,:]

            RAIN=torch.mul(PRECIP,(TEMP>=parTT).type(torch.float32))
            SNOW=torch.mul(PRECIP,(TEMP<parTT).type(torch.float32))

            # Snow process
            SNOWPACK=SNOWPACK+SNOW
            melt=parCFMAX*(TEMP-parTT)
            melt=torch.clamp(melt,min=0.0)
            melt=torch.min(melt,SNOWPACK)
            MELTWATER=MELTWATER+melt
            SNOWPACK=SNOWPACK-melt
            refreezing=parCFR*parCFMAX*(parTT-TEMP)
            refreezing=torch.clamp(refreezing,min=0.0)
            refreezing=torch.min(refreezing,MELTWATER)
            SNOWPACK=SNOWPACK+refreezing
            MELTWATER=MELTWATER-refreezing
            tosoil=MELTWATER-(parCWH*SNOWPACK)
            tosoil=torch.clamp(tosoil,min=0.0)
            MELTWATER=MELTWATER-tosoil

            # Soil and evaporation
            soil_wetness=(SM/parFC)**parBETA
            soil_wetness=torch.clamp(soil_wetness,min=0.0,max=1.0)
            recharge=(RAIN+tosoil)*soil_wetness

            # Not used, logging states for checking
            # logSM[t,:] = SM.detach().cpu().numpy()
            # logPS[t,:] = (RAIN + tosoil).detach().cpu().numpy()
            # logswet[t,:] = (SM / parFC).detach().cpu().numpy()
            # logRE[t, :] = recharge.detach().cpu().numpy()

            SM=SM+RAIN+tosoil-recharge
            excess=SM-parFC
            excess=torch.clamp(excess,min=0.0)
            SM=SM-excess
            # MODIFY here. Different as class HBVMulT(). Add a ET shape parameter parBETAET
            evapfactor=(SM/(parLP*parFC))**parBETAET
            evapfactor=torch.clamp(evapfactor,min=0.0,max=1.0)
            ETact=EVAP_P*evapfactor
            ETact=torch.min(SM,ETact)
            SM=torch.clamp(SM-ETact,min=PRECS)  # SM can not be zero for gradient tracking

            capillary=torch.min(SLZ,parC*SLZ*(1.0-torch.clamp(SM/parFC,max=1.0)))

            SM=torch.clamp(SM+capillary,min=PRECS)
            SLZ=torch.clamp(SLZ-capillary,min=PRECS)

            # Groundwater boxes
            SUZ=SUZ+recharge+excess
            PERC=torch.min(SUZ,parPERC)
            SUZ=SUZ-PERC
            Q0=parK0*torch.clamp(SUZ-parUZL,min=0.0)
            SUZ=SUZ-Q0
            Q1=parK1*SUZ
            SUZ=SUZ-Q1
            SLZ=SLZ+PERC
            Q2=parK2*SLZ
            SLZ=SLZ-Q2
            Qsimmu[t,:,:]=Q0+Q1+Q2

            # save components for Q
            Qsimmu0[t,:,:]=Q0
            Qsimmu1[t,:,:]=Q1
            Qsimmu2[t,:,:]=Q2
            ETmu[t,:,:]=ETact
            SWEmu[t,:,:]=SNOWPACK
            # Not used, for debug state variables
            # SMlog[t,:, :] = SM.detach().cpu().numpy()
            # SUZlog[t,:,:] = SUZ.detach().cpu().numpy()
            # SLZlog[t,:,:] = SLZ.detach().cpu().numpy()

        Qsimave0=Qsimmu0.mean(-1,keepdim=True)
        Qsimave1=Qsimmu1.mean(-1,keepdim=True)
        Qsimave2=Qsimmu2.mean(-1,keepdim=True)
        ETave=ETmu.mean(-1,keepdim=True)
        SWEave=SWEmu.mean(-1,keepdim=True)

        # get the initial average
        if muwts is None:  # simple average
            Qsimave=Qsimmu.mean(-1)
        else:  # weighted average using learned weights
            Qsimave=(Qsimmu*muwts).sum(-1)

        if routOpt is True:  # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim=Qsimmu.view(Nstep,Ngrid*mu)
            else:
                # average the components, then do routing
                Qsim=Qsimave

            # scale two routing parameters
            tempa=routscaLst[0][0]+rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb=routscaLst[1][0]+rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])
            routa=tempa.repeat(Nstep,1).unsqueeze(-1)
            routb=tempb.repeat(Nstep,1).unsqueeze(-1)
            UH=UH_gamma(routa,routb,lenF=15)  # lenF: folter
            rf=torch.unsqueeze(Qsim,-1).permute([1,2,0])  # dim:gage*var*time
            UH=UH.permute([1,2,0])  # dim: gage*var*time
            Qsrout=UH_conv(rf,UH).permute([2,0,1])

            if comprout is True:  # Qs is [time, [gage*mult], var] now
                Qstemp=Qsrout.view(Nstep,Ngrid,mu)
                if muwts is None:
                    Qs=Qstemp.mean(-1,keepdim=True)
                else:
                    Qs=(Qstemp*muwts).sum(-1,keepdim=True)
            else:
                Qs=Qsrout

        else:  # no routing, output the initial average simulations
            Qs=torch.unsqueeze(Qsimave,-1)  # add a dimension

        if outstate is True:  # output states
            return Qs,SNOWPACK,MELTWATER,SM,SUZ,SLZ
        else:
            # return Qs
            Qall=torch.cat((Qs,Qsimave0,Qsimave1,Qsimave2,ETave,SWEave),dim=-1)
            return Qall

class HBVMulTDET_1_1p_PE_bias3(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""

    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first
    # only P bais, k and t

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMulTDET_1_1p_PE_bias3,self).__init__()

    def forward(self,x,parameters,staind,tdlst,mu,muwts,rtwts,bufftime=0,outstate=False,routOpt=False,
                comprout=False,dydrop=False):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     X: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2]
        #     staind:use which time step from the learned para time series for static parameters
        #     tdlst: the index list of hbv parameters set as dynamic
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     dydrop: the possibility to drop a dynamic para to static to reduce potential overfitting
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET

        PRECS=1e-5  # keep the numerical calculation stable
        # bufftime=0 # pjli 20240926, donot understand why, comment it
        # Initialization to warm-up states
        if bufftime>0:
            with torch.no_grad():
                xinit=x[0:bufftime,:,:]
                initmodel=HBVMulTDET_1_1p()
                # buffpara=parameters[bufftime-1,:,:,:] # pjli 20240926 original
                buffpara = parameters[0:bufftime,:,:] # pql5336, dynamic parameter warm up
                Qsinit,SNOWPACK,MELTWATER,SM,SUZ,SLZ=initmodel(xinit,buffpara,staind,tdlst,mu,muwts,rtwts,
                                                               bufftime=0,outstate=True,routOpt=False,comprout=False,
                                                               dydrop=False)
        else:

            # Without warm-up bufftime=0, initialize state variables with zeros
            Ngrid=x.shape[1]
            SNOWPACK=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            MELTWATER=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            SM=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            SUZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            SLZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(x)

        P=x[bufftime:,:,0]
        Pm=P.unsqueeze(2).repeat(1,1,mu)  # precip
        T=x[bufftime:,:,1]
        Tm=T.unsqueeze(2).repeat(1,1,mu)  # temperature
        ETpot=x[bufftime:,:,2]
        ETpm=ETpot.unsqueeze(2).repeat(1,1,mu)  # potential ET
        parAll=parameters[bufftime:,:,:,:]
        parAllTrans=torch.zeros_like(parAll)

        ## scale the parameters to real values from [0,1]
        hbvscaLst=[[1,6],[50,1000],[0.05,0.9],[0.01,0.5],[0.001,0.2],[0.2,1],
                   [0,10],[0,100],[-2.5,2.5],[0.5,10],[0,0.1],[0,0.2],[0.3,5],[0,1],[-0.1,0.1],[-10,10]]  # HBV para
        routscaLst=[[0,2.9],[0,6.5]]  # routing para

        for ip in range(len(hbvscaLst)):  # not include routing. Scaling the parameters
            parAllTrans[:,:,ip,:]=hbvscaLst[ip][0]+parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        Nstep,Ngrid=P.size()

        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        parstaFull=parAllTrans[staind,:,:,:].unsqueeze(0).repeat([Nstep,1,1,1])  # static para matrix
        parhbvFull=torch.clone(parstaFull)
        # create probability mask for each parameter on the basin dimension
        pmat=torch.ones([1,Ngrid,1])*dydrop
        for ix in tdlst:
            staPar=parstaFull[:,:,ix-1,:]
            dynPar=parAllTrans[:,:,ix-1,:]
            drmask=torch.bernoulli(pmat).detach_().to(parhbvFull)  # to drop dynamic parameters as static in some basins
            comPar=dynPar*(1-drmask)+staPar*drmask
            parhbvFull[:,:,ix-1,:]=comPar

        # Initialize time series of model variables to save results
        Qsimmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        ETmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        SWEmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        # Output the box components of Q
        Qsimmu0=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu1=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu2=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)

        # # Not used. Logging the state variables for debug.
        # # SMlog = np.zeros(P.size())
        # logSM = np.zeros(P.size())
        # logPS = np.zeros(P.size())
        # logswet = np.zeros(P.size())
        # logRE = np.zeros(P.size())

        for t in range(Nstep):
            paraLst=[]
            for ip in range(len(hbvscaLst)):  # unpack HBV parameters
                paraLst.append(parhbvFull[t,:,ip,:])

            parBETA,parFC,parK0,parK1,parK2,parLP,parPERC,parUZL,parTT,parCFMAX,parCFR,parCWH,parBETAET,parC, P_bias, P_bias_t=paraLst
            # Separate precipitation into liquid and solid components
            PRECIP=Pm[t,:,:]+P_bias*Pm[t,:,:]+P_bias_t
            PRECIP=torch.clamp(PRECIP,min=0.0)
            TEMP=Tm[t,:,:]
            EVAP_P=ETpm[t,:,:]

            RAIN=torch.mul(PRECIP,(TEMP>=parTT).type(torch.float32))
            SNOW=torch.mul(PRECIP,(TEMP<parTT).type(torch.float32))

            # Snow process
            SNOWPACK=SNOWPACK+SNOW
            melt=parCFMAX*(TEMP-parTT)
            melt=torch.clamp(melt,min=0.0)
            melt=torch.min(melt,SNOWPACK)
            MELTWATER=MELTWATER+melt
            SNOWPACK=SNOWPACK-melt
            refreezing=parCFR*parCFMAX*(parTT-TEMP)
            refreezing=torch.clamp(refreezing,min=0.0)
            refreezing=torch.min(refreezing,MELTWATER)
            SNOWPACK=SNOWPACK+refreezing
            MELTWATER=MELTWATER-refreezing
            tosoil=MELTWATER-(parCWH*SNOWPACK)
            tosoil=torch.clamp(tosoil,min=0.0)
            MELTWATER=MELTWATER-tosoil

            # Soil and evaporation
            soil_wetness=(SM/parFC)**parBETA
            soil_wetness=torch.clamp(soil_wetness,min=0.0,max=1.0)
            recharge=(RAIN+tosoil)*soil_wetness

            # Not used, logging states for checking
            # logSM[t,:] = SM.detach().cpu().numpy()
            # logPS[t,:] = (RAIN + tosoil).detach().cpu().numpy()
            # logswet[t,:] = (SM / parFC).detach().cpu().numpy()
            # logRE[t, :] = recharge.detach().cpu().numpy()

            SM=SM+RAIN+tosoil-recharge
            excess=SM-parFC
            excess=torch.clamp(excess,min=0.0)
            SM=SM-excess
            # MODIFY here. Different as class HBVMulT(). Add a ET shape parameter parBETAET
            evapfactor=(SM/(parLP*parFC))**parBETAET
            evapfactor=torch.clamp(evapfactor,min=0.0,max=1.0)
            ETact=EVAP_P*evapfactor
            ETact=torch.min(SM,ETact)
            SM=torch.clamp(SM-ETact,min=PRECS)  # SM can not be zero for gradient tracking

            capillary=torch.min(SLZ,parC*SLZ*(1.0-torch.clamp(SM/parFC,max=1.0)))

            SM=torch.clamp(SM+capillary,min=PRECS)
            SLZ=torch.clamp(SLZ-capillary,min=PRECS)

            # Groundwater boxes
            SUZ=SUZ+recharge+excess
            PERC=torch.min(SUZ,parPERC)
            SUZ=SUZ-PERC
            Q0=parK0*torch.clamp(SUZ-parUZL,min=0.0)
            SUZ=SUZ-Q0
            Q1=parK1*SUZ
            SUZ=SUZ-Q1
            SLZ=SLZ+PERC
            Q2=parK2*SLZ
            SLZ=SLZ-Q2
            Qsimmu[t,:,:]=Q0+Q1+Q2

            # save components for Q
            Qsimmu0[t,:,:]=Q0
            Qsimmu1[t,:,:]=Q1
            Qsimmu2[t,:,:]=Q2
            ETmu[t,:,:]=ETact
            SWEmu[t,:,:]=SNOWPACK
            # Not used, for debug state variables
            # SMlog[t,:, :] = SM.detach().cpu().numpy()
            # SUZlog[t,:,:] = SUZ.detach().cpu().numpy()
            # SLZlog[t,:,:] = SLZ.detach().cpu().numpy()

        Qsimave0=Qsimmu0.mean(-1,keepdim=True)
        Qsimave1=Qsimmu1.mean(-1,keepdim=True)
        Qsimave2=Qsimmu2.mean(-1,keepdim=True)
        ETave=ETmu.mean(-1,keepdim=True)
        SWEave=SWEmu.mean(-1,keepdim=True)

        # get the initial average
        if muwts is None:  # simple average
            Qsimave=Qsimmu.mean(-1)
        else:  # weighted average using learned weights
            Qsimave=(Qsimmu*muwts).sum(-1)

        if routOpt is True:  # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim=Qsimmu.view(Nstep,Ngrid*mu)
            else:
                # average the components, then do routing
                Qsim=Qsimave

            # scale two routing parameters
            tempa=routscaLst[0][0]+rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb=routscaLst[1][0]+rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])
            routa=tempa.repeat(Nstep,1).unsqueeze(-1)
            routb=tempb.repeat(Nstep,1).unsqueeze(-1)
            UH=UH_gamma(routa,routb,lenF=15)  # lenF: folter
            rf=torch.unsqueeze(Qsim,-1).permute([1,2,0])  # dim:gage*var*time
            UH=UH.permute([1,2,0])  # dim: gage*var*time
            Qsrout=UH_conv(rf,UH).permute([2,0,1])

            if comprout is True:  # Qs is [time, [gage*mult], var] now
                Qstemp=Qsrout.view(Nstep,Ngrid,mu)
                if muwts is None:
                    Qs=Qstemp.mean(-1,keepdim=True)
                else:
                    Qs=(Qstemp*muwts).sum(-1,keepdim=True)
            else:
                Qs=Qsrout

        else:  # no routing, output the initial average simulations
            Qs=torch.unsqueeze(Qsimave,-1)  # add a dimension

        if outstate is True:  # output states
            return Qs,SNOWPACK,MELTWATER,SM,SUZ,SLZ
        else:
            # return Qs
            Qall=torch.cat((Qs,Qsimave0,Qsimave1,Qsimave2,ETave,SWEave),dim=-1)
            return Qall

class HBVMulTDET_1_1p_PE_bias4(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""

    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first
    # only P bias, k and t, ET bias, k and t

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMulTDET_1_1p_PE_bias4,self).__init__()

    def forward(self,x,parameters,staind,tdlst,mu,muwts,rtwts,bufftime=0,outstate=False,routOpt=False,
                comprout=False,dydrop=False):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     X: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2]
        #     staind:use which time step from the learned para time series for static parameters
        #     tdlst: the index list of hbv parameters set as dynamic
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     dydrop: the possibility to drop a dynamic para to static to reduce potential overfitting
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET

        PRECS=1e-5  # keep the numerical calculation stable
        # bufftime=0 # pjli 20240926, donot understand why, comment it
        # Initialization to warm-up states
        if bufftime>0:
            with torch.no_grad():
                xinit=x[0:bufftime,:,:]
                initmodel=HBVMulTDET_1_1p()
                # buffpara=parameters[bufftime-1,:,:,:] # pjli 20240926 original
                buffpara = parameters[0:bufftime,:,:] # pql5336, dynamic parameter warm up
                Qsinit,SNOWPACK,MELTWATER,SM,SUZ,SLZ=initmodel(xinit,buffpara,staind,tdlst,mu,muwts,rtwts,
                                                               bufftime=0,outstate=True,routOpt=False,comprout=False,
                                                               dydrop=False)
        else:

            # Without warm-up bufftime=0, initialize state variables with zeros
            Ngrid=x.shape[1]
            SNOWPACK=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            MELTWATER=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            SM=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            SUZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            SLZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(x)

        P=x[bufftime:,:,0]
        Pm=P.unsqueeze(2).repeat(1,1,mu)  # precip
        T=x[bufftime:,:,1]
        Tm=T.unsqueeze(2).repeat(1,1,mu)  # temperature
        ETpot=x[bufftime:,:,2]
        ETpm=ETpot.unsqueeze(2).repeat(1,1,mu)  # potential ET
        parAll=parameters[bufftime:,:,:,:]
        parAllTrans=torch.zeros_like(parAll)

        ## scale the parameters to real values from [0,1]
        hbvscaLst=[[1,6],[50,1000],[0.05,0.9],[0.01,0.5],[0.001,0.2],[0.2,1],
                   [0,10],[0,100],[-2.5,2.5],[0.5,10],[0,0.1],[0,0.2],[0.3,5],[0,1],[-0.1,0.1],[-10,10],[-0.1,0.1],[-10,10]]  # HBV para
        routscaLst=[[0,2.9],[0,6.5]]  # routing para

        for ip in range(len(hbvscaLst)):  # not include routing. Scaling the parameters
            parAllTrans[:,:,ip,:]=hbvscaLst[ip][0]+parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        Nstep,Ngrid=P.size()

        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        parstaFull=parAllTrans[staind,:,:,:].unsqueeze(0).repeat([Nstep,1,1,1])  # static para matrix
        parhbvFull=torch.clone(parstaFull)
        # create probability mask for each parameter on the basin dimension
        pmat=torch.ones([1,Ngrid,1])*dydrop
        for ix in tdlst:
            staPar=parstaFull[:,:,ix-1,:]
            dynPar=parAllTrans[:,:,ix-1,:]
            drmask=torch.bernoulli(pmat).detach_().to(parhbvFull)  # to drop dynamic parameters as static in some basins
            comPar=dynPar*(1-drmask)+staPar*drmask
            parhbvFull[:,:,ix-1,:]=comPar

        # Initialize time series of model variables to save results
        Qsimmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        ETmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        SWEmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        # Output the box components of Q
        Qsimmu0=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu1=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu2=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)

        # # Not used. Logging the state variables for debug.
        # # SMlog = np.zeros(P.size())
        # logSM = np.zeros(P.size())
        # logPS = np.zeros(P.size())
        # logswet = np.zeros(P.size())
        # logRE = np.zeros(P.size())

        for t in range(Nstep):
            paraLst=[]
            for ip in range(len(hbvscaLst)):  # unpack HBV parameters
                paraLst.append(parhbvFull[t,:,ip,:])

            parBETA,parFC,parK0,parK1,parK2,parLP,parPERC,parUZL,parTT,parCFMAX,parCFR,parCWH,parBETAET,parC, P_bias, P_bias_t, E_bias, E_bias_t=paraLst
            # Separate precipitation into liquid and solid components
            PRECIP=Pm[t,:,:]+P_bias*Pm[t,:,:]+P_bias_t
            PRECIP=torch.clamp(PRECIP,min=0.0)
            TEMP=Tm[t,:,:]
            EVAP_P=ETpm[t,:,:]+E_bias*ETpm[t,:,:]+E_bias_t
            EVAP_P=torch.clamp(EVAP_P, min=0.0)

            RAIN=torch.mul(PRECIP,(TEMP>=parTT).type(torch.float32))
            SNOW=torch.mul(PRECIP,(TEMP<parTT).type(torch.float32))

            # Snow process
            SNOWPACK=SNOWPACK+SNOW
            melt=parCFMAX*(TEMP-parTT)
            melt=torch.clamp(melt,min=0.0)
            melt=torch.min(melt,SNOWPACK)
            MELTWATER=MELTWATER+melt
            SNOWPACK=SNOWPACK-melt
            refreezing=parCFR*parCFMAX*(parTT-TEMP)
            refreezing=torch.clamp(refreezing,min=0.0)
            refreezing=torch.min(refreezing,MELTWATER)
            SNOWPACK=SNOWPACK+refreezing
            MELTWATER=MELTWATER-refreezing
            tosoil=MELTWATER-(parCWH*SNOWPACK)
            tosoil=torch.clamp(tosoil,min=0.0)
            MELTWATER=MELTWATER-tosoil

            # Soil and evaporation
            soil_wetness=(SM/parFC)**parBETA
            soil_wetness=torch.clamp(soil_wetness,min=0.0,max=1.0)
            recharge=(RAIN+tosoil)*soil_wetness

            # Not used, logging states for checking
            # logSM[t,:] = SM.detach().cpu().numpy()
            # logPS[t,:] = (RAIN + tosoil).detach().cpu().numpy()
            # logswet[t,:] = (SM / parFC).detach().cpu().numpy()
            # logRE[t, :] = recharge.detach().cpu().numpy()

            SM=SM+RAIN+tosoil-recharge
            excess=SM-parFC
            excess=torch.clamp(excess,min=0.0)
            SM=SM-excess
            # MODIFY here. Different as class HBVMulT(). Add a ET shape parameter parBETAET
            evapfactor=(SM/(parLP*parFC))**parBETAET
            evapfactor=torch.clamp(evapfactor,min=0.0,max=1.0)
            ETact=EVAP_P*evapfactor
            ETact=torch.min(SM,ETact)
            SM=torch.clamp(SM-ETact,min=PRECS)  # SM can not be zero for gradient tracking

            capillary=torch.min(SLZ,parC*SLZ*(1.0-torch.clamp(SM/parFC,max=1.0)))

            SM=torch.clamp(SM+capillary,min=PRECS)
            SLZ=torch.clamp(SLZ-capillary,min=PRECS)

            # Groundwater boxes
            SUZ=SUZ+recharge+excess
            PERC=torch.min(SUZ,parPERC)
            SUZ=SUZ-PERC
            Q0=parK0*torch.clamp(SUZ-parUZL,min=0.0)
            SUZ=SUZ-Q0
            Q1=parK1*SUZ
            SUZ=SUZ-Q1
            SLZ=SLZ+PERC
            Q2=parK2*SLZ
            SLZ=SLZ-Q2
            Qsimmu[t,:,:]=Q0+Q1+Q2

            # save components for Q
            Qsimmu0[t,:,:]=Q0
            Qsimmu1[t,:,:]=Q1
            Qsimmu2[t,:,:]=Q2
            ETmu[t,:,:]=ETact
            SWEmu[t,:,:]=SNOWPACK
            # Not used, for debug state variables
            # SMlog[t,:, :] = SM.detach().cpu().numpy()
            # SUZlog[t,:,:] = SUZ.detach().cpu().numpy()
            # SLZlog[t,:,:] = SLZ.detach().cpu().numpy()

        Qsimave0=Qsimmu0.mean(-1,keepdim=True)
        Qsimave1=Qsimmu1.mean(-1,keepdim=True)
        Qsimave2=Qsimmu2.mean(-1,keepdim=True)
        ETave=ETmu.mean(-1,keepdim=True)
        SWEave=SWEmu.mean(-1,keepdim=True)

        # get the initial average
        if muwts is None:  # simple average
            Qsimave=Qsimmu.mean(-1)
        else:  # weighted average using learned weights
            Qsimave=(Qsimmu*muwts).sum(-1)

        if routOpt is True:  # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim=Qsimmu.view(Nstep,Ngrid*mu)
            else:
                # average the components, then do routing
                Qsim=Qsimave

            # scale two routing parameters
            tempa=routscaLst[0][0]+rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb=routscaLst[1][0]+rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])
            routa=tempa.repeat(Nstep,1).unsqueeze(-1)
            routb=tempb.repeat(Nstep,1).unsqueeze(-1)
            UH=UH_gamma(routa,routb,lenF=15)  # lenF: folter
            rf=torch.unsqueeze(Qsim,-1).permute([1,2,0])  # dim:gage*var*time
            UH=UH.permute([1,2,0])  # dim: gage*var*time
            Qsrout=UH_conv(rf,UH).permute([2,0,1])

            if comprout is True:  # Qs is [time, [gage*mult], var] now
                Qstemp=Qsrout.view(Nstep,Ngrid,mu)
                if muwts is None:
                    Qs=Qstemp.mean(-1,keepdim=True)
                else:
                    Qs=(Qstemp*muwts).sum(-1,keepdim=True)
            else:
                Qs=Qsrout

        else:  # no routing, output the initial average simulations
            Qs=torch.unsqueeze(Qsimave,-1)  # add a dimension

        if outstate is True:  # output states
            return Qs,SNOWPACK,MELTWATER,SM,SUZ,SLZ
        else:
            # return Qs
            Qall=torch.cat((Qs,Qsimave0,Qsimave1,Qsimave2,ETave,SWEave),dim=-1)
            return Qall


# Create a polynomial evaluation function using broadcasting
def batch_poly_eval(x,coeffs):
    # x: [10, 16]
    # coeffs: [10, 6]

    # Create polynomial powers [5, 4, 3, 2, 1, 0]
    powers=torch.arange(coeffs.size(1)-1,-1,-1).unsqueeze(0).unsqueeze(2).to(x)  # [1, 6, 1]

    # Evaluate the polynomial for each feature value of x
    # Use broadcasting to expand x and powers so they can be multiplied elementwise
    x_expanded=x.unsqueeze(1)  # [10, 1, 16]
    x_powers=x_expanded**powers  # [10, 6, 16]，different powers of x

    # Apply the polynomial coefficients to the different powers of x and sum the polynomial terms for each sample
    result=torch.sum(coeffs*x_powers,dim=1)  # [10, 16]

    return result

class HBVMulTDET_1_1p_HAND(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""

    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMulTDET_1_1p_HAND,self).__init__()
        self.HSC = AnnModel_static(nx=5+1,ny=1,hiddenSize=30)


    def forward(self,x,zd_feature,parameters,staind,tdlst,mu,muwts,rtwts,bufftime=0,outstate=False,routOpt=False,
                comprout=False,dydrop=False):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     X: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2]
        #     staind:use which time step from the learned para time series for static parameters
        #     tdlst: the index list of hbv parameters set as dynamic
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     dydrop: the possibility to drop a dynamic para to static to reduce potential overfitting
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET

        PRECS=1e-5  # keep the numerical calculation stable
        # bufftime=0 # pjli 20240926, donot understand why, comment it
        # Initialization to warm-up states
        if bufftime>0:
            with torch.no_grad():
                xinit=x[0:bufftime,:,:]
                initmodel=HBVMulTDET_1_1p_HAND()
                # buffpara=parameters[bufftime-1,:,:,:] # pjli 20240926 original
                buffpara = parameters[0:bufftime,:,:] # pql5336, dynamic parameter warm up
                Qsinit,SNOWPACK,MELTWATER,SM,SUZ,SLZ=initmodel(xinit,zd_feature,buffpara,staind,tdlst,mu,muwts,rtwts,
                                                               bufftime=0,outstate=True,routOpt=False,comprout=False,
                                                               dydrop=False)
        else:

            # Without warm-up bufftime=0, initialize state variables with zeros
            Ngrid=x.shape[1]
            SNOWPACK=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            MELTWATER=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            SM=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            SUZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            SLZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(x)

        P=x[bufftime:,:,0]
        Pm=P.unsqueeze(2).repeat(1,1,mu)  # precip
        T=x[bufftime:,:,1]
        Tm=T.unsqueeze(2).repeat(1,1,mu)  # temperature
        ETpot=x[bufftime:,:,2]
        ETpm=ETpot.unsqueeze(2).repeat(1,1,mu)  # potential ET
        parAll=parameters[bufftime:,:,:,:]
        parAllTrans=torch.zeros_like(parAll)

        ## scale the parameters to real values from [0,1]
        hbvscaLst=[[0.5,1.5],[50,1000],[0.05,0.9],[0.01,0.5],[0.001,0.2],[0.2,1],
                   [0,10],[0,100],[-2.5,2.5],[0.5,10],[0,0.1],[0,0.2],[0.3,5],[0,1]]  # HBV para
        routscaLst=[[0,2.9],[0,6.5]]  # routing para

        for ip in range(len(hbvscaLst)):  # not include routing. Scaling the parameters
            parAllTrans[:,:,ip,:]=hbvscaLst[ip][0]+parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        Nstep,Ngrid=P.size()

        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        parstaFull=parAllTrans[staind,:,:,:].unsqueeze(0).repeat([Nstep,1,1,1])  # static para matrix
        parhbvFull=torch.clone(parstaFull)
        # create probability mask for each parameter on the basin dimension
        pmat=torch.ones([1,Ngrid,1])*dydrop
        for ix in tdlst:
            staPar=parstaFull[:,:,ix-1,:]
            dynPar=parAllTrans[:,:,ix-1,:]
            drmask=torch.bernoulli(pmat).detach_().to(parhbvFull)  # to drop dynamic parameters as static in some basins
            comPar=dynPar*(1-drmask)+staPar*drmask
            parhbvFull[:,:,ix-1,:]=comPar

        # Initialize time series of model variables to save results
        Qsimmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        ETmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        SWEmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        # Output the box components of Q
        Qsimmu0=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu1=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu2=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)

        # # Not used. Logging the state variables for debug.
        # # SMlog = np.zeros(P.size())
        # logSM = np.zeros(P.size())
        # logPS = np.zeros(P.size())
        # logswet = np.zeros(P.size())
        # logRE = np.zeros(P.size())

        for t in range(Nstep):
            paraLst=[]
            for ip in range(len(hbvscaLst)):  # unpack HBV parameters
                paraLst.append(parhbvFull[t,:,ip,:])

            parBETA,parFC,parK0,parK1,parK2,parLP,parPERC,parUZL,parTT,parCFMAX,parCFR,parCWH,parBETAET,parC=paraLst
            # Separate precipitation into liquid and solid components
            PRECIP=Pm[t,:,:]
            TEMP=Tm[t,:,:]
            EVAP_P=ETpm[t,:,:]

            RAIN=torch.mul(PRECIP,(TEMP>=parTT).type(torch.float32))
            SNOW=torch.mul(PRECIP,(TEMP<parTT).type(torch.float32))

            # Snow process
            SNOWPACK=SNOWPACK+SNOW
            melt=parCFMAX*(TEMP-parTT)
            melt=torch.clamp(melt,min=0.0)
            melt=torch.min(melt,SNOWPACK)
            MELTWATER=MELTWATER+melt
            SNOWPACK=SNOWPACK-melt
            refreezing=parCFR*parCFMAX*(parTT-TEMP)
            refreezing=torch.clamp(refreezing,min=0.0)
            refreezing=torch.min(refreezing,MELTWATER)
            SNOWPACK=SNOWPACK+refreezing
            MELTWATER=MELTWATER-refreezing
            tosoil=MELTWATER-(parCWH*SNOWPACK)
            tosoil=torch.clamp(tosoil,min=0.0)
            MELTWATER=MELTWATER-tosoil

            # Soil and evaporation
            # soil_wetness=(SM/parFC)**parBETA
            # soil_wetness=self.HSC(zd_feature,SM/parFC)
            soil_wetness=batch_poly_eval(SM/parFC,zd_feature[:,:,None])
            soil_wetness=torch.clamp(soil_wetness,min=0.0,max=1.0)
            recharge=(RAIN+tosoil)*soil_wetness

            # Not used, logging states for checking
            # logSM[t,:] = SM.detach().cpu().numpy()
            # logPS[t,:] = (RAIN + tosoil).detach().cpu().numpy()
            # logswet[t,:] = (SM / parFC).detach().cpu().numpy()
            # logRE[t, :] = recharge.detach().cpu().numpy()

            SM=SM+RAIN+tosoil-recharge
            excess=SM-parFC
            excess=torch.clamp(excess,min=0.0)
            SM=SM-excess
            # MODIFY here. Different as class HBVMulT(). Add a ET shape parameter parBETAET
            evapfactor=(SM/(parLP*parFC))**parBETAET
            evapfactor=torch.clamp(evapfactor,min=0.0,max=1.0)
            ETact=EVAP_P*evapfactor
            ETact=torch.min(SM,ETact)
            SM=torch.clamp(SM-ETact,min=PRECS)  # SM can not be zero for gradient tracking

            capillary=torch.min(SLZ,parC*SLZ*(1.0-torch.clamp(SM/parFC,max=1.0)))

            SM=torch.clamp(SM+capillary,min=PRECS)
            SLZ=torch.clamp(SLZ-capillary,min=PRECS)

            # Groundwater boxes
            SUZ=SUZ+recharge+excess
            PERC=torch.min(SUZ,parPERC)
            SUZ=SUZ-PERC
            Q0=parK0*torch.clamp(SUZ-parUZL,min=0.0)
            SUZ=SUZ-Q0
            Q1=parK1*SUZ
            SUZ=SUZ-Q1
            SLZ=SLZ+PERC
            Q2=parK2*SLZ
            SLZ=SLZ-Q2
            Qsimmu[t,:,:]=Q0+Q1+Q2

            # save components for Q
            Qsimmu0[t,:,:]=Q0
            Qsimmu1[t,:,:]=Q1
            Qsimmu2[t,:,:]=Q2
            ETmu[t,:,:]=ETact
            SWEmu[t,:,:]=SNOWPACK
            # Not used, for debug state variables
            # SMlog[t,:, :] = SM.detach().cpu().numpy()
            # SUZlog[t,:,:] = SUZ.detach().cpu().numpy()
            # SLZlog[t,:,:] = SLZ.detach().cpu().numpy()

        Qsimave0=Qsimmu0.mean(-1,keepdim=True)
        Qsimave1=Qsimmu1.mean(-1,keepdim=True)
        Qsimave2=Qsimmu2.mean(-1,keepdim=True)
        ETave=ETmu.mean(-1,keepdim=True)
        SWEave=SWEmu.mean(-1,keepdim=True)

        # get the initial average
        if muwts is None:  # simple average
            Qsimave=Qsimmu.mean(-1)
        else:  # weighted average using learned weights
            Qsimave=(Qsimmu*muwts).sum(-1)

        if routOpt is True:  # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim=Qsimmu.view(Nstep,Ngrid*mu)
            else:
                # average the components, then do routing
                Qsim=Qsimave

            # scale two routing parameters
            tempa=routscaLst[0][0]+rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb=routscaLst[1][0]+rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])
            routa=tempa.repeat(Nstep,1).unsqueeze(-1)
            routb=tempb.repeat(Nstep,1).unsqueeze(-1)
            UH=UH_gamma(routa,routb,lenF=15)  # lenF: folter
            rf=torch.unsqueeze(Qsim,-1).permute([1,2,0])  # dim:gage*var*time
            UH=UH.permute([1,2,0])  # dim: gage*var*time
            Qsrout=UH_conv(rf,UH).permute([2,0,1])

            if comprout is True:  # Qs is [time, [gage*mult], var] now
                Qstemp=Qsrout.view(Nstep,Ngrid,mu)
                if muwts is None:
                    Qs=Qstemp.mean(-1,keepdim=True)
                else:
                    Qs=(Qstemp*muwts).sum(-1,keepdim=True)
            else:
                Qs=Qsrout

        else:  # no routing, output the initial average simulations
            Qs=torch.unsqueeze(Qsimave,-1)  # add a dimension

        if outstate is True:  # output states
            return Qs,SNOWPACK,MELTWATER,SM,SUZ,SLZ
        else:
            # return Qs
            Qall=torch.cat((Qs,Qsimave0,Qsimave1,Qsimave2,ETave,SWEave),dim=-1)
            return Qall


class HBVMulTDET_1_1p_HAND_beta(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""

    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first
    # no estimate coeff, but estimate a dyn beta to scale the coeff

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMulTDET_1_1p_HAND_beta,self).__init__()
        self.HSC = AnnModel_static(nx=5+1,ny=1,hiddenSize=30)


    def forward(self,x,zd_feature,parameters,staind,tdlst,mu,muwts,rtwts,bufftime=0,outstate=False,routOpt=False,
                comprout=False,dydrop=False):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     X: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2]
        #     staind:use which time step from the learned para time series for static parameters
        #     tdlst: the index list of hbv parameters set as dynamic
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     dydrop: the possibility to drop a dynamic para to static to reduce potential overfitting
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET

        PRECS=1e-5  # keep the numerical calculation stable
        # bufftime=0 # pjli 20240926, donot understand why, comment it
        # Initialization to warm-up states
        if bufftime>0:
            with torch.no_grad():
                xinit=x[0:bufftime,:,:]
                initmodel=HBVMulTDET_1_1p_HAND_beta()
                # buffpara=parameters[bufftime-1,:,:,:] # pjli 20240926 original
                buffpara = parameters[0:bufftime,:,:] # pql5336, dynamic parameter warm up
                Qsinit,SNOWPACK,MELTWATER,SM,SUZ,SLZ=initmodel(xinit,zd_feature,buffpara,staind,tdlst,mu,muwts,rtwts,
                                                               bufftime=0,outstate=True,routOpt=False,comprout=False,
                                                               dydrop=False)
        else:

            # Without warm-up bufftime=0, initialize state variables with zeros
            Ngrid=x.shape[1]
            SNOWPACK=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            MELTWATER=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            SM=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            SUZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            SLZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(x)

        P=x[bufftime:,:,0]
        Pm=P.unsqueeze(2).repeat(1,1,mu)  # precip
        T=x[bufftime:,:,1]
        Tm=T.unsqueeze(2).repeat(1,1,mu)  # temperature
        ETpot=x[bufftime:,:,2]
        ETpm=ETpot.unsqueeze(2).repeat(1,1,mu)  # potential ET
        parAll=parameters[bufftime:,:,:,:]
        parAllTrans=torch.zeros_like(parAll)

        ## scale the parameters to real values from [0,1]
        hbvscaLst=[[0.5,1.5],[50,1000],[0.05,0.9],[0.01,0.5],[0.001,0.2],[0.2,1],
                   [0,10],[0,100],[-2.5,2.5],[0.5,10],[0,0.1],[0,0.2],[0.3,5],[0,1]]  # HBV para
        routscaLst=[[0,2.9],[0,6.5]]  # routing para

        for ip in range(len(hbvscaLst)):  # not include routing. Scaling the parameters
            parAllTrans[:,:,ip,:]=hbvscaLst[ip][0]+parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        Nstep,Ngrid=P.size()

        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        parstaFull=parAllTrans[staind,:,:,:].unsqueeze(0).repeat([Nstep,1,1,1])  # static para matrix
        parhbvFull=torch.clone(parstaFull)
        # create probability mask for each parameter on the basin dimension
        pmat=torch.ones([1,Ngrid,1])*dydrop
        for ix in tdlst:
            staPar=parstaFull[:,:,ix-1,:]
            dynPar=parAllTrans[:,:,ix-1,:]
            drmask=torch.bernoulli(pmat).detach_().to(parhbvFull)  # to drop dynamic parameters as static in some basins
            comPar=dynPar*(1-drmask)+staPar*drmask
            parhbvFull[:,:,ix-1,:]=comPar

        # Initialize time series of model variables to save results
        Qsimmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        ETmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        SWEmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        # Output the box components of Q
        Qsimmu0=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu1=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu2=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)

        # # Not used. Logging the state variables for debug.
        # # SMlog = np.zeros(P.size())
        # logSM = np.zeros(P.size())
        # logPS = np.zeros(P.size())
        # logswet = np.zeros(P.size())
        # logRE = np.zeros(P.size())

        for t in range(Nstep):
            paraLst=[]
            for ip in range(len(hbvscaLst)):  # unpack HBV parameters
                paraLst.append(parhbvFull[t,:,ip,:])

            parBETA,parFC,parK0,parK1,parK2,parLP,parPERC,parUZL,parTT,parCFMAX,parCFR,parCWH,parBETAET,parC=paraLst
            # Separate precipitation into liquid and solid components
            PRECIP=Pm[t,:,:]
            TEMP=Tm[t,:,:]
            EVAP_P=ETpm[t,:,:]

            RAIN=torch.mul(PRECIP,(TEMP>=parTT).type(torch.float32))
            SNOW=torch.mul(PRECIP,(TEMP<parTT).type(torch.float32))

            # Snow process
            SNOWPACK=SNOWPACK+SNOW
            melt=parCFMAX*(TEMP-parTT)
            melt=torch.clamp(melt,min=0.0)
            melt=torch.min(melt,SNOWPACK)
            MELTWATER=MELTWATER+melt
            SNOWPACK=SNOWPACK-melt
            refreezing=parCFR*parCFMAX*(parTT-TEMP)
            refreezing=torch.clamp(refreezing,min=0.0)
            refreezing=torch.min(refreezing,MELTWATER)
            SNOWPACK=SNOWPACK+refreezing
            MELTWATER=MELTWATER-refreezing
            tosoil=MELTWATER-(parCWH*SNOWPACK)
            tosoil=torch.clamp(tosoil,min=0.0)
            MELTWATER=MELTWATER-tosoil

            # Soil and evaporation
            # soil_wetness=(SM/parFC)**parBETA
            # soil_wetness=self.HSC(zd_feature,SM/parFC)
            soil_wetness=batch_poly_eval(SM/parFC,zd_feature[:,:,None]*parBETA[:,None])
            soil_wetness=torch.clamp(soil_wetness,min=0.0,max=1.0)
            recharge=(RAIN+tosoil)*soil_wetness

            # Not used, logging states for checking
            # logSM[t,:] = SM.detach().cpu().numpy()
            # logPS[t,:] = (RAIN + tosoil).detach().cpu().numpy()
            # logswet[t,:] = (SM / parFC).detach().cpu().numpy()
            # logRE[t, :] = recharge.detach().cpu().numpy()

            SM=SM+RAIN+tosoil-recharge
            excess=SM-parFC
            excess=torch.clamp(excess,min=0.0)
            SM=SM-excess
            # MODIFY here. Different as class HBVMulT(). Add a ET shape parameter parBETAET
            evapfactor=(SM/(parLP*parFC))**parBETAET
            evapfactor=torch.clamp(evapfactor,min=0.0,max=1.0)
            ETact=EVAP_P*evapfactor
            ETact=torch.min(SM,ETact)
            SM=torch.clamp(SM-ETact,min=PRECS)  # SM can not be zero for gradient tracking

            capillary=torch.min(SLZ,parC*SLZ*(1.0-torch.clamp(SM/parFC,max=1.0)))

            SM=torch.clamp(SM+capillary,min=PRECS)
            SLZ=torch.clamp(SLZ-capillary,min=PRECS)

            # Groundwater boxes
            SUZ=SUZ+recharge+excess
            PERC=torch.min(SUZ,parPERC)
            SUZ=SUZ-PERC
            Q0=parK0*torch.clamp(SUZ-parUZL,min=0.0)
            SUZ=SUZ-Q0
            Q1=parK1*SUZ
            SUZ=SUZ-Q1
            SLZ=SLZ+PERC
            Q2=parK2*SLZ
            SLZ=SLZ-Q2
            Qsimmu[t,:,:]=Q0+Q1+Q2

            # save components for Q
            Qsimmu0[t,:,:]=Q0
            Qsimmu1[t,:,:]=Q1
            Qsimmu2[t,:,:]=Q2
            ETmu[t,:,:]=ETact
            SWEmu[t,:,:]=SNOWPACK
            # Not used, for debug state variables
            # SMlog[t,:, :] = SM.detach().cpu().numpy()
            # SUZlog[t,:,:] = SUZ.detach().cpu().numpy()
            # SLZlog[t,:,:] = SLZ.detach().cpu().numpy()

        Qsimave0=Qsimmu0.mean(-1,keepdim=True)
        Qsimave1=Qsimmu1.mean(-1,keepdim=True)
        Qsimave2=Qsimmu2.mean(-1,keepdim=True)
        ETave=ETmu.mean(-1,keepdim=True)
        SWEave=SWEmu.mean(-1,keepdim=True)

        # get the initial average
        if muwts is None:  # simple average
            Qsimave=Qsimmu.mean(-1)
        else:  # weighted average using learned weights
            Qsimave=(Qsimmu*muwts).sum(-1)

        if routOpt is True:  # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim=Qsimmu.view(Nstep,Ngrid*mu)
            else:
                # average the components, then do routing
                Qsim=Qsimave

            # scale two routing parameters
            tempa=routscaLst[0][0]+rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb=routscaLst[1][0]+rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])
            routa=tempa.repeat(Nstep,1).unsqueeze(-1)
            routb=tempb.repeat(Nstep,1).unsqueeze(-1)
            UH=UH_gamma(routa,routb,lenF=15)  # lenF: folter
            rf=torch.unsqueeze(Qsim,-1).permute([1,2,0])  # dim:gage*var*time
            UH=UH.permute([1,2,0])  # dim: gage*var*time
            Qsrout=UH_conv(rf,UH).permute([2,0,1])

            if comprout is True:  # Qs is [time, [gage*mult], var] now
                Qstemp=Qsrout.view(Nstep,Ngrid,mu)
                if muwts is None:
                    Qs=Qstemp.mean(-1,keepdim=True)
                else:
                    Qs=(Qstemp*muwts).sum(-1,keepdim=True)
            else:
                Qs=Qsrout

        else:  # no routing, output the initial average simulations
            Qs=torch.unsqueeze(Qsimave,-1)  # add a dimension

        if outstate is True:  # output states
            return Qs,SNOWPACK,MELTWATER,SM,SUZ,SLZ
        else:
            # return Qs
            Qall=torch.cat((Qs,Qsimave0,Qsimave1,Qsimave2,ETave,SWEave),dim=-1)
            return Qall

class HBVMulTDET_1_1p_HAND_coeff(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""

    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first
    # 6 poly coefficiences are estimated by NN

    def __init__(self,nmul):
        """Initiate a HBV instance"""
        super(HBVMulTDET_1_1p_HAND_coeff,self).__init__()
        self.HSC = AnnModel_static(nx=nmul*7,ny=nmul,hiddenSize=60)


    def forward(self,x,parameters,para_coeff, staind,tdlst,mu,muwts,rtwts,bufftime=0,outstate=False,routOpt=False,
                comprout=False,dydrop=False):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     X: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2]
        #     staind:use which time step from the learned para time series for static parameters
        #     tdlst: the index list of hbv parameters set as dynamic
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     dydrop: the possibility to drop a dynamic para to static to reduce potential overfitting
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET

        PRECS=1e-5  # keep the numerical calculation stable
        # bufftime=0 # pjli 20240926, donot understand why, comment it
        # Initialization to warm-up states
        if bufftime>0:
            with torch.no_grad():
                xinit=x[0:bufftime,:,:]
                initmodel=HBVMulTDET_1_1p_HAND_coeff(nmul)
                # buffpara=parameters[bufftime-1,:,:,:] # pjli 20240926 original
                buffpara = parameters[0:bufftime,:,:] # pql5336, dynamic parameter warm up
                buff_para_coeff = para_coeff[0:bufftime,:,:]
                Qsinit,SNOWPACK,MELTWATER,SM,SUZ,SLZ=initmodel(xinit,buffpara,buff_para_coeff,staind,tdlst,mu,muwts,rtwts,
                                                               bufftime=0,outstate=True,routOpt=False,comprout=False,
                                                               dydrop=False)
        else:

            # Without warm-up bufftime=0, initialize state variables with zeros
            Ngrid=x.shape[1]
            SNOWPACK=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            MELTWATER=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            SM=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            SUZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            SLZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(x)

        P=x[bufftime:,:,0]
        Pm=P.unsqueeze(2).repeat(1,1,mu)  # precip
        T=x[bufftime:,:,1]
        Tm=T.unsqueeze(2).repeat(1,1,mu)  # temperature
        ETpot=x[bufftime:,:,2]
        ETpm=ETpot.unsqueeze(2).repeat(1,1,mu)  # potential ET
        parAll=parameters[bufftime:,:,:,:]
        parAllTrans=torch.zeros_like(parAll)

        ## scale the parameters to real values from [0,1]
        hbvscaLst=[[0.5,1.5],[50,1000],[0.05,0.9],[0.01,0.5],[0.001,0.2],[0.2,1],
                   [0,10],[0,100],[-2.5,2.5],[0.5,10],[0,0.1],[0,0.2],[0.3,5],[0,1]]  # HBV para
        routscaLst=[[0,2.9],[0,6.5]]  # routing para

        for ip in range(len(hbvscaLst)):  # not include routing. Scaling the parameters
            parAllTrans[:,:,ip,:]=hbvscaLst[ip][0]+parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        Nstep,Ngrid=P.size()

        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        parstaFull=parAllTrans[staind,:,:,:].unsqueeze(0).repeat([Nstep,1,1,1])  # static para matrix
        parhbvFull=torch.clone(parstaFull)
        # create probability mask for each parameter on the basin dimension
        pmat=torch.ones([1,Ngrid,1])*dydrop
        for ix in tdlst:
            staPar=parstaFull[:,:,ix-1,:]
            dynPar=parAllTrans[:,:,ix-1,:]
            drmask=torch.bernoulli(pmat).detach_().to(parhbvFull)  # to drop dynamic parameters as static in some basins
            comPar=dynPar*(1-drmask)+staPar*drmask
            parhbvFull[:,:,ix-1,:]=comPar

        # Initialize time series of model variables to save results
        Qsimmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        ETmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        SWEmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        # Output the box components of Q
        Qsimmu0=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu1=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu2=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)

        # # Not used. Logging the state variables for debug.
        # # SMlog = np.zeros(P.size())
        # logSM = np.zeros(P.size())
        # logPS = np.zeros(P.size())
        # logswet = np.zeros(P.size())
        # logRE = np.zeros(P.size())

        for t in range(Nstep):
            paraLst=[]

            zd_feature = para_coeff[t+bufftime]

            for ip in range(len(hbvscaLst)):  # unpack HBV parameters
                paraLst.append(parhbvFull[t,:,ip,:])

            parBETA,parFC,parK0,parK1,parK2,parLP,parPERC,parUZL,parTT,parCFMAX,parCFR,parCWH,parBETAET,parC=paraLst
            # Separate precipitation into liquid and solid components
            PRECIP=Pm[t,:,:]
            TEMP=Tm[t,:,:]
            EVAP_P=ETpm[t,:,:]

            RAIN=torch.mul(PRECIP,(TEMP>=parTT).type(torch.float32))
            SNOW=torch.mul(PRECIP,(TEMP<parTT).type(torch.float32))

            # Snow process
            SNOWPACK=SNOWPACK+SNOW
            melt=parCFMAX*(TEMP-parTT)
            melt=torch.clamp(melt,min=0.0)
            melt=torch.min(melt,SNOWPACK)
            MELTWATER=MELTWATER+melt
            SNOWPACK=SNOWPACK-melt
            refreezing=parCFR*parCFMAX*(parTT-TEMP)
            refreezing=torch.clamp(refreezing,min=0.0)
            refreezing=torch.min(refreezing,MELTWATER)
            SNOWPACK=SNOWPACK+refreezing
            MELTWATER=MELTWATER-refreezing
            tosoil=MELTWATER-(parCWH*SNOWPACK)
            tosoil=torch.clamp(tosoil,min=0.0)
            MELTWATER=MELTWATER-tosoil

            # Soil and evaporation
            # soil_wetness=(SM/parFC)**parBETA
            # soil_wetness=self.HSC(zd_feature,SM/parFC)
            soil_wetness=batch_poly_eval(SM/parFC,zd_feature[:,:,:])
            soil_wetness=torch.clamp(soil_wetness,min=0.0,max=1.0)
            recharge=(RAIN+tosoil)*soil_wetness

            # Not used, logging states for checking
            # logSM[t,:] = SM.detach().cpu().numpy()
            # logPS[t,:] = (RAIN + tosoil).detach().cpu().numpy()
            # logswet[t,:] = (SM / parFC).detach().cpu().numpy()
            # logRE[t, :] = recharge.detach().cpu().numpy()

            SM=SM+RAIN+tosoil-recharge
            excess=SM-parFC
            excess=torch.clamp(excess,min=0.0)
            SM=SM-excess
            # MODIFY here. Different as class HBVMulT(). Add a ET shape parameter parBETAET
            evapfactor=(SM/(parLP*parFC))**parBETAET
            evapfactor=torch.clamp(evapfactor,min=0.0,max=1.0)
            ETact=EVAP_P*evapfactor
            ETact=torch.min(SM,ETact)
            SM=torch.clamp(SM-ETact,min=PRECS)  # SM can not be zero for gradient tracking

            capillary=torch.min(SLZ,parC*SLZ*(1.0-torch.clamp(SM/parFC,max=1.0)))

            SM=torch.clamp(SM+capillary,min=PRECS)
            SLZ=torch.clamp(SLZ-capillary,min=PRECS)

            # Groundwater boxes
            SUZ=SUZ+recharge+excess
            PERC=torch.min(SUZ,parPERC)
            SUZ=SUZ-PERC
            Q0=parK0*torch.clamp(SUZ-parUZL,min=0.0)
            SUZ=SUZ-Q0
            Q1=parK1*SUZ
            SUZ=SUZ-Q1
            SLZ=SLZ+PERC
            Q2=parK2*SLZ
            SLZ=SLZ-Q2
            Qsimmu[t,:,:]=Q0+Q1+Q2

            # save components for Q
            Qsimmu0[t,:,:]=Q0
            Qsimmu1[t,:,:]=Q1
            Qsimmu2[t,:,:]=Q2
            ETmu[t,:,:]=ETact
            SWEmu[t,:,:]=SNOWPACK
            # Not used, for debug state variables
            # SMlog[t,:, :] = SM.detach().cpu().numpy()
            # SUZlog[t,:,:] = SUZ.detach().cpu().numpy()
            # SLZlog[t,:,:] = SLZ.detach().cpu().numpy()

        Qsimave0=Qsimmu0.mean(-1,keepdim=True)
        Qsimave1=Qsimmu1.mean(-1,keepdim=True)
        Qsimave2=Qsimmu2.mean(-1,keepdim=True)
        ETave=ETmu.mean(-1,keepdim=True)
        SWEave=SWEmu.mean(-1,keepdim=True)

        # get the initial average
        if muwts is None:  # simple average
            Qsimave=Qsimmu.mean(-1)
        else:  # weighted average using learned weights
            Qsimave=(Qsimmu*muwts).sum(-1)

        if routOpt is True:  # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim=Qsimmu.view(Nstep,Ngrid*mu)
            else:
                # average the components, then do routing
                Qsim=Qsimave

            # scale two routing parameters
            tempa=routscaLst[0][0]+rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb=routscaLst[1][0]+rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])
            routa=tempa.repeat(Nstep,1).unsqueeze(-1)
            routb=tempb.repeat(Nstep,1).unsqueeze(-1)
            UH=UH_gamma(routa,routb,lenF=15)  # lenF: folter
            rf=torch.unsqueeze(Qsim,-1).permute([1,2,0])  # dim:gage*var*time
            UH=UH.permute([1,2,0])  # dim: gage*var*time
            Qsrout=UH_conv(rf,UH).permute([2,0,1])

            if comprout is True:  # Qs is [time, [gage*mult], var] now
                Qstemp=Qsrout.view(Nstep,Ngrid,mu)
                if muwts is None:
                    Qs=Qstemp.mean(-1,keepdim=True)
                else:
                    Qs=(Qstemp*muwts).sum(-1,keepdim=True)
            else:
                Qs=Qsrout

        else:  # no routing, output the initial average simulations
            Qs=torch.unsqueeze(Qsimave,-1)  # add a dimension

        if outstate is True:  # output states
            return Qs,SNOWPACK,MELTWATER,SM,SUZ,SLZ
        else:
            # return Qs
            Qall=torch.cat((Qs,Qsimave0,Qsimave1,Qsimave2,ETave,SWEave),dim=-1)
            return Qall

class HBVMulTDET_1_1p_HAND_coeff_NN(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""

    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first
    # 6 coefficiences are estimated by NN
    # HSC is estimated by NN

    def __init__(self,nmul):
        """Initiate a HBV instance"""
        super(HBVMulTDET_1_1p_HAND_coeff_NN,self).__init__()
        self.nmul=nmul
        self.HSC = AnnModel_static(nx=self.nmul*7,ny=self.nmul,hiddenSize=60)


    def forward(self,x,parameters,para_coeff, staind,tdlst,mu,muwts,rtwts,bufftime=0,outstate=False,routOpt=False,
                comprout=False,dydrop=False):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     X: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2]
        #     staind:use which time step from the learned para time series for static parameters
        #     tdlst: the index list of hbv parameters set as dynamic
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     dydrop: the possibility to drop a dynamic para to static to reduce potential overfitting
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET

        PRECS=1e-5  # keep the numerical calculation stable
        # bufftime=0 # pjli 20240926, donot understand why, comment it
        # Initialization to warm-up states
        if bufftime>0:
            with torch.no_grad():
                xinit=x[0:bufftime,:,:]
                initmodel=HBVMulTDET_1_1p_HAND_coeff_NN(self.nmul).to(xinit)
                # buffpara=parameters[bufftime-1,:,:,:] # pjli 20240926 original
                buffpara = parameters[0:bufftime,:,:] # pql5336, dynamic parameter warm up
                buff_para_coeff=para_coeff[0:bufftime,:,:]
                Qsinit,SNOWPACK,MELTWATER,SM,SUZ,SLZ=initmodel(xinit,buffpara,buff_para_coeff,staind,tdlst,mu,muwts,rtwts,
                                                               bufftime=0,outstate=True,routOpt=False,comprout=False,
                                                               dydrop=False)
        else:

            # Without warm-up bufftime=0, initialize state variables with zeros
            Ngrid=x.shape[1]
            SNOWPACK=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            MELTWATER=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            SM=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            SUZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            SLZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(x)

        P=x[bufftime:,:,0]
        Pm=P.unsqueeze(2).repeat(1,1,mu)  # precip
        T=x[bufftime:,:,1]
        Tm=T.unsqueeze(2).repeat(1,1,mu)  # temperature
        ETpot=x[bufftime:,:,2]
        ETpm=ETpot.unsqueeze(2).repeat(1,1,mu)  # potential ET
        parAll=parameters[bufftime:,:,:,:]
        parAllTrans=torch.zeros_like(parAll)

        ## scale the parameters to real values from [0,1]
        hbvscaLst=[[0.5,1.5],[50,1000],[0.05,0.9],[0.01,0.5],[0.001,0.2],[0.2,1],
                   [0,10],[0,100],[-2.5,2.5],[0.5,10],[0,0.1],[0,0.2],[0.3,5],[0,1]]  # HBV para
        routscaLst=[[0,2.9],[0,6.5]]  # routing para

        for ip in range(len(hbvscaLst)):  # not include routing. Scaling the parameters
            parAllTrans[:,:,ip,:]=hbvscaLst[ip][0]+parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        Nstep,Ngrid=P.size()

        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        parstaFull=parAllTrans[staind,:,:,:].unsqueeze(0).repeat([Nstep,1,1,1])  # static para matrix
        parhbvFull=torch.clone(parstaFull)
        # create probability mask for each parameter on the basin dimension
        pmat=torch.ones([1,Ngrid,1])*dydrop
        for ix in tdlst:
            staPar=parstaFull[:,:,ix-1,:]
            dynPar=parAllTrans[:,:,ix-1,:]
            drmask=torch.bernoulli(pmat).detach_().to(parhbvFull)  # to drop dynamic parameters as static in some basins
            comPar=dynPar*(1-drmask)+staPar*drmask
            parhbvFull[:,:,ix-1,:]=comPar

        # Initialize time series of model variables to save results
        Qsimmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        ETmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        SWEmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        # Output the box components of Q
        Qsimmu0=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu1=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu2=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)

        # # Not used. Logging the state variables for debug.
        # # SMlog = np.zeros(P.size())
        # logSM = np.zeros(P.size())
        # logPS = np.zeros(P.size())
        # logswet = np.zeros(P.size())
        # logRE = np.zeros(P.size())

        for t in range(Nstep):
            paraLst=[]

            zd_feature = para_coeff[t+bufftime]

            for ip in range(len(hbvscaLst)):  # unpack HBV parameters
                paraLst.append(parhbvFull[t,:,ip,:])

            parBETA,parFC,parK0,parK1,parK2,parLP,parPERC,parUZL,parTT,parCFMAX,parCFR,parCWH,parBETAET,parC=paraLst
            # Separate precipitation into liquid and solid components
            PRECIP=Pm[t,:,:]
            TEMP=Tm[t,:,:]
            EVAP_P=ETpm[t,:,:]

            RAIN=torch.mul(PRECIP,(TEMP>=parTT).type(torch.float32))
            SNOW=torch.mul(PRECIP,(TEMP<parTT).type(torch.float32))

            # Snow process
            SNOWPACK=SNOWPACK+SNOW
            melt=parCFMAX*(TEMP-parTT)
            melt=torch.clamp(melt,min=0.0)
            melt=torch.min(melt,SNOWPACK)
            MELTWATER=MELTWATER+melt
            SNOWPACK=SNOWPACK-melt
            refreezing=parCFR*parCFMAX*(parTT-TEMP)
            refreezing=torch.clamp(refreezing,min=0.0)
            refreezing=torch.min(refreezing,MELTWATER)
            SNOWPACK=SNOWPACK+refreezing
            MELTWATER=MELTWATER-refreezing
            tosoil=MELTWATER-(parCWH*SNOWPACK)
            tosoil=torch.clamp(tosoil,min=0.0)
            MELTWATER=MELTWATER-tosoil

            # Soil and evaporation
            # soil_wetness=(SM/parFC)**parBETA
            # soil_wetness=self.HSC(zd_feature,SM/parFC)
            # soil_wetness=batch_poly_eval(SM/parFC,zd_feature[:,:,:])
            HSC_input_ori = torch.cat(((SM/parFC)[:,None],zd_feature),dim=1)
            batch_size,height,width=HSC_input_ori.shape
            HSC_input = HSC_input_ori.view(batch_size, height*width)
            soil_wetness = torch.sigmoid(self.HSC(HSC_input))
            soil_wetness=torch.clamp(soil_wetness,min=0.0,max=1.0)
            recharge=(RAIN+tosoil)*soil_wetness

            # Not used, logging states for checking
            # logSM[t,:] = SM.detach().cpu().numpy()
            # logPS[t,:] = (RAIN + tosoil).detach().cpu().numpy()
            # logswet[t,:] = (SM / parFC).detach().cpu().numpy()
            # logRE[t, :] = recharge.detach().cpu().numpy()

            SM=SM+RAIN+tosoil-recharge
            excess=SM-parFC
            excess=torch.clamp(excess,min=0.0)
            SM=SM-excess
            # MODIFY here. Different as class HBVMulT(). Add a ET shape parameter parBETAET
            evapfactor=(SM/(parLP*parFC))**parBETAET
            evapfactor=torch.clamp(evapfactor,min=0.0,max=1.0)
            ETact=EVAP_P*evapfactor
            ETact=torch.min(SM,ETact)
            SM=torch.clamp(SM-ETact,min=PRECS)  # SM can not be zero for gradient tracking

            capillary=torch.min(SLZ,parC*SLZ*(1.0-torch.clamp(SM/parFC,max=1.0)))

            SM=torch.clamp(SM+capillary,min=PRECS)
            SLZ=torch.clamp(SLZ-capillary,min=PRECS)

            # Groundwater boxes
            SUZ=SUZ+recharge+excess
            PERC=torch.min(SUZ,parPERC)
            SUZ=SUZ-PERC
            Q0=parK0*torch.clamp(SUZ-parUZL,min=0.0)
            SUZ=SUZ-Q0
            Q1=parK1*SUZ
            SUZ=SUZ-Q1
            SLZ=SLZ+PERC
            Q2=parK2*SLZ
            SLZ=SLZ-Q2
            Qsimmu[t,:,:]=Q0+Q1+Q2

            # save components for Q
            Qsimmu0[t,:,:]=Q0
            Qsimmu1[t,:,:]=Q1
            Qsimmu2[t,:,:]=Q2
            ETmu[t,:,:]=ETact
            SWEmu[t,:,:]=SNOWPACK
            # Not used, for debug state variables
            # SMlog[t,:, :] = SM.detach().cpu().numpy()
            # SUZlog[t,:,:] = SUZ.detach().cpu().numpy()
            # SLZlog[t,:,:] = SLZ.detach().cpu().numpy()

        Qsimave0=Qsimmu0.mean(-1,keepdim=True)
        Qsimave1=Qsimmu1.mean(-1,keepdim=True)
        Qsimave2=Qsimmu2.mean(-1,keepdim=True)
        ETave=ETmu.mean(-1,keepdim=True)
        SWEave=SWEmu.mean(-1,keepdim=True)

        # get the initial average
        if muwts is None:  # simple average
            Qsimave=Qsimmu.mean(-1)
        else:  # weighted average using learned weights
            Qsimave=(Qsimmu*muwts).sum(-1)

        if routOpt is True:  # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim=Qsimmu.view(Nstep,Ngrid*mu)
            else:
                # average the components, then do routing
                Qsim=Qsimave

            # scale two routing parameters
            tempa=routscaLst[0][0]+rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb=routscaLst[1][0]+rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])
            routa=tempa.repeat(Nstep,1).unsqueeze(-1)
            routb=tempb.repeat(Nstep,1).unsqueeze(-1)
            UH=UH_gamma(routa,routb,lenF=15)  # lenF: folter
            rf=torch.unsqueeze(Qsim,-1).permute([1,2,0])  # dim:gage*var*time
            UH=UH.permute([1,2,0])  # dim: gage*var*time
            Qsrout=UH_conv(rf,UH).permute([2,0,1])

            if comprout is True:  # Qs is [time, [gage*mult], var] now
                Qstemp=Qsrout.view(Nstep,Ngrid,mu)
                if muwts is None:
                    Qs=Qstemp.mean(-1,keepdim=True)
                else:
                    Qs=(Qstemp*muwts).sum(-1,keepdim=True)
            else:
                Qs=Qsrout

        else:  # no routing, output the initial average simulations
            Qs=torch.unsqueeze(Qsimave,-1)  # add a dimension

        if outstate is True:  # output states
            return Qs,SNOWPACK,MELTWATER,SM,SUZ,SLZ
        else:
            # return Qs
            Qall=torch.cat((Qs,Qsimave0,Qsimave1,Qsimave2,ETave,SWEave),dim=-1)
            return Qall

class HBVMulTDET_1_1p_HAND_coeff_NN_morefeature(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""

    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first
    # 6 coefficiences are estimated by NN
    # HSC is estimated by NN

    def __init__(self,nmul):
        """Initiate a HBV instance"""
        super(HBVMulTDET_1_1p_HAND_coeff_NN_morefeature,self).__init__()
        self.HSC = AnnModel_static(nx=nmul*(10+1),ny=nmul,hiddenSize=60)


    def forward(self,x,parameters,para_coeff, staind,tdlst,mu,muwts,rtwts,bufftime=0,outstate=False,routOpt=False,
                comprout=False,dydrop=False):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     X: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2]
        #     staind:use which time step from the learned para time series for static parameters
        #     tdlst: the index list of hbv parameters set as dynamic
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     dydrop: the possibility to drop a dynamic para to static to reduce potential overfitting
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET

        PRECS=1e-5  # keep the numerical calculation stable
        # bufftime=0 # pjli 20240926, donot understand why, comment it
        # Initialization to warm-up states
        if bufftime>0:
            with torch.no_grad():
                xinit=x[0:bufftime,:,:]
                initmodel=HBVMulTDET_1_1p_HAND_coeff_NN_morefeature(nmul)
                # buffpara=parameters[bufftime-1,:,:,:] # pjli 20240926 original
                buffpara = parameters[0:bufftime,:,:] # pql5336, dynamic parameter warm up
                buff_para_coeff = para_coeff[0:bufftime,:,:]
                Qsinit,SNOWPACK,MELTWATER,SM,SUZ,SLZ=initmodel(xinit,buffpara,buff_para_coeff,staind,tdlst,mu,muwts,rtwts,
                                                               bufftime=0,outstate=True,routOpt=False,comprout=False,
                                                               dydrop=False)
        else:

            # Without warm-up bufftime=0, initialize state variables with zeros
            Ngrid=x.shape[1]
            SNOWPACK=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            MELTWATER=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            SM=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            SUZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            SLZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
            # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(x)

        P=x[bufftime:,:,0]
        Pm=P.unsqueeze(2).repeat(1,1,mu)  # precip
        T=x[bufftime:,:,1]
        Tm=T.unsqueeze(2).repeat(1,1,mu)  # temperature
        ETpot=x[bufftime:,:,2]
        ETpm=ETpot.unsqueeze(2).repeat(1,1,mu)  # potential ET
        parAll=parameters[bufftime:,:,:,:]
        parAllTrans=torch.zeros_like(parAll)

        ## scale the parameters to real values from [0,1]
        hbvscaLst=[[0.5,1.5],[50,1000],[0.05,0.9],[0.01,0.5],[0.001,0.2],[0.2,1],
                   [0,10],[0,100],[-2.5,2.5],[0.5,10],[0,0.1],[0,0.2],[0.3,5],[0,1]]  # HBV para
        routscaLst=[[0,2.9],[0,6.5]]  # routing para

        for ip in range(len(hbvscaLst)):  # not include routing. Scaling the parameters
            parAllTrans[:,:,ip,:]=hbvscaLst[ip][0]+parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        Nstep,Ngrid=P.size()

        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        parstaFull=parAllTrans[staind,:,:,:].unsqueeze(0).repeat([Nstep,1,1,1])  # static para matrix
        parhbvFull=torch.clone(parstaFull)
        # create probability mask for each parameter on the basin dimension
        pmat=torch.ones([1,Ngrid,1])*dydrop
        for ix in tdlst:
            staPar=parstaFull[:,:,ix-1,:]
            dynPar=parAllTrans[:,:,ix-1,:]
            drmask=torch.bernoulli(pmat).detach_().to(parhbvFull)  # to drop dynamic parameters as static in some basins
            comPar=dynPar*(1-drmask)+staPar*drmask
            parhbvFull[:,:,ix-1,:]=comPar

        # Initialize time series of model variables to save results
        Qsimmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        ETmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        SWEmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        # Output the box components of Q
        Qsimmu0=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu1=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu2=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)

        # # Not used. Logging the state variables for debug.
        # # SMlog = np.zeros(P.size())
        # logSM = np.zeros(P.size())
        # logPS = np.zeros(P.size())
        # logswet = np.zeros(P.size())
        # logRE = np.zeros(P.size())

        for t in range(Nstep):
            paraLst=[]

            zd_feature = para_coeff[t+bufftime]

            for ip in range(len(hbvscaLst)):  # unpack HBV parameters
                paraLst.append(parhbvFull[t,:,ip,:])

            parBETA,parFC,parK0,parK1,parK2,parLP,parPERC,parUZL,parTT,parCFMAX,parCFR,parCWH,parBETAET,parC=paraLst
            # Separate precipitation into liquid and solid components
            PRECIP=Pm[t,:,:]
            TEMP=Tm[t,:,:]
            EVAP_P=ETpm[t,:,:]

            RAIN=torch.mul(PRECIP,(TEMP>=parTT).type(torch.float32))
            SNOW=torch.mul(PRECIP,(TEMP<parTT).type(torch.float32))

            # Snow process
            SNOWPACK=SNOWPACK+SNOW
            melt=parCFMAX*(TEMP-parTT)
            melt=torch.clamp(melt,min=0.0)
            melt=torch.min(melt,SNOWPACK)
            MELTWATER=MELTWATER+melt
            SNOWPACK=SNOWPACK-melt
            refreezing=parCFR*parCFMAX*(parTT-TEMP)
            refreezing=torch.clamp(refreezing,min=0.0)
            refreezing=torch.min(refreezing,MELTWATER)
            SNOWPACK=SNOWPACK+refreezing
            MELTWATER=MELTWATER-refreezing
            tosoil=MELTWATER-(parCWH*SNOWPACK)
            tosoil=torch.clamp(tosoil,min=0.0)
            MELTWATER=MELTWATER-tosoil

            # Soil and evaporation
            # soil_wetness=(SM/parFC)**parBETA
            # soil_wetness=self.HSC(zd_feature,SM/parFC)
            # soil_wetness=batch_poly_eval(SM/parFC,zd_feature[:,:,:])
            HSC_input_ori = torch.cat(((SM/parFC)[:,None],zd_feature),dim=1)
            batch_size,height,width=HSC_input_ori.shape
            HSC_input = HSC_input_ori.view(batch_size, height*width)
            soil_wetness = torch.sigmoid(self.HSC(HSC_input))
            soil_wetness=torch.clamp(soil_wetness,min=0.0,max=1.0)
            recharge=(RAIN+tosoil)*soil_wetness

            # Not used, logging states for checking
            # logSM[t,:] = SM.detach().cpu().numpy()
            # logPS[t,:] = (RAIN + tosoil).detach().cpu().numpy()
            # logswet[t,:] = (SM / parFC).detach().cpu().numpy()
            # logRE[t, :] = recharge.detach().cpu().numpy()

            SM=SM+RAIN+tosoil-recharge
            excess=SM-parFC
            excess=torch.clamp(excess,min=0.0)
            SM=SM-excess
            # MODIFY here. Different as class HBVMulT(). Add a ET shape parameter parBETAET
            evapfactor=(SM/(parLP*parFC))**parBETAET
            evapfactor=torch.clamp(evapfactor,min=0.0,max=1.0)
            ETact=EVAP_P*evapfactor
            ETact=torch.min(SM,ETact)
            SM=torch.clamp(SM-ETact,min=PRECS)  # SM can not be zero for gradient tracking

            capillary=torch.min(SLZ,parC*SLZ*(1.0-torch.clamp(SM/parFC,max=1.0)))

            SM=torch.clamp(SM+capillary,min=PRECS)
            SLZ=torch.clamp(SLZ-capillary,min=PRECS)

            # Groundwater boxes
            SUZ=SUZ+recharge+excess
            PERC=torch.min(SUZ,parPERC)
            SUZ=SUZ-PERC
            Q0=parK0*torch.clamp(SUZ-parUZL,min=0.0)
            SUZ=SUZ-Q0
            Q1=parK1*SUZ
            SUZ=SUZ-Q1
            SLZ=SLZ+PERC
            Q2=parK2*SLZ
            SLZ=SLZ-Q2
            Qsimmu[t,:,:]=Q0+Q1+Q2

            # save components for Q
            Qsimmu0[t,:,:]=Q0
            Qsimmu1[t,:,:]=Q1
            Qsimmu2[t,:,:]=Q2
            ETmu[t,:,:]=ETact
            SWEmu[t,:,:]=SNOWPACK
            # Not used, for debug state variables
            # SMlog[t,:, :] = SM.detach().cpu().numpy()
            # SUZlog[t,:,:] = SUZ.detach().cpu().numpy()
            # SLZlog[t,:,:] = SLZ.detach().cpu().numpy()

        Qsimave0=Qsimmu0.mean(-1,keepdim=True)
        Qsimave1=Qsimmu1.mean(-1,keepdim=True)
        Qsimave2=Qsimmu2.mean(-1,keepdim=True)
        ETave=ETmu.mean(-1,keepdim=True)
        SWEave=SWEmu.mean(-1,keepdim=True)

        # get the initial average
        if muwts is None:  # simple average
            Qsimave=Qsimmu.mean(-1)
        else:  # weighted average using learned weights
            Qsimave=(Qsimmu*muwts).sum(-1)

        if routOpt is True:  # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim=Qsimmu.view(Nstep,Ngrid*mu)
            else:
                # average the components, then do routing
                Qsim=Qsimave

            # scale two routing parameters
            tempa=routscaLst[0][0]+rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb=routscaLst[1][0]+rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])
            routa=tempa.repeat(Nstep,1).unsqueeze(-1)
            routb=tempb.repeat(Nstep,1).unsqueeze(-1)
            UH=UH_gamma(routa,routb,lenF=15)  # lenF: folter
            rf=torch.unsqueeze(Qsim,-1).permute([1,2,0])  # dim:gage*var*time
            UH=UH.permute([1,2,0])  # dim: gage*var*time
            Qsrout=UH_conv(rf,UH).permute([2,0,1])

            if comprout is True:  # Qs is [time, [gage*mult], var] now
                Qstemp=Qsrout.view(Nstep,Ngrid,mu)
                if muwts is None:
                    Qs=Qstemp.mean(-1,keepdim=True)
                else:
                    Qs=(Qstemp*muwts).sum(-1,keepdim=True)
            else:
                Qs=Qsrout

        else:  # no routing, output the initial average simulations
            Qs=torch.unsqueeze(Qsimave,-1)  # add a dimension

        if outstate is True:  # output states
            return Qs,SNOWPACK,MELTWATER,SM,SUZ,SLZ
        else:
            # return Qs
            Qall=torch.cat((Qs,Qsimave0,Qsimave1,Qsimave2,ETave,SWEave),dim=-1)
            return Qall


class HBVMulTDET_1_1p_wateruse(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""

    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first
    # 6 coefficiences are estimated by NN
    # HSC is estimated by NN

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMulTDET_1_1p_wateruse,self).__init__()


    def forward(self,x,parameters, wateruse, staind,tdlst,mu,muwts,rtwts,bufftime=0,outstate=False,routOpt=False,
                comprout=False,dydrop=False):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     X: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2]
        #     staind:use which time step from the learned para time series for static parameters
        #     tdlst: the index list of hbv parameters set as dynamic
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     dydrop: the possibility to drop a dynamic para to static to reduce potential overfitting
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET

        PRECS=1e-5  # keep the numerical calculation stable
        bufftime_old = bufftime
        bufftime=0 # pjli 20240926, donot understand why, comment it
        # # Initialization to warm-up states
        # if bufftime>0:
        #     with torch.no_grad():
        #         xinit=x[0:bufftime,:,:]
        #         initmodel=HBVMulTDET_1_1p_wateruse()
        #         # buffpara=parameters[bufftime-1,:,:,:] # pjli 20240926 original
        #         buffpara = parameters[0:bufftime,:,:] # pql5336, dynamic parameter warm up
        #         Qsinit,SNOWPACK,MELTWATER,SM,SUZ,SLZ=initmodel(xinit,buffpara,wateruse[0:bufftime],staind,tdlst,mu,muwts,rtwts,
        #                                                        bufftime=0,outstate=True,routOpt=False,comprout=False,
        #                                                        dydrop=False)
        # else:

        # Without warm-up bufftime=0, initialize state variables with zeros
        Ngrid=x.shape[1]
        SNOWPACK=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        MELTWATER=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SM=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SUZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SLZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(x)

        P=x[bufftime:,:,0]
        Pm=P.unsqueeze(2).repeat(1,1,mu)  # precip
        T=x[bufftime:,:,1]
        Tm=T.unsqueeze(2).repeat(1,1,mu)  # temperature
        ETpot=x[bufftime:,:,2]
        ETpm=ETpot.unsqueeze(2).repeat(1,1,mu)  # potential ET
        parAll=parameters[bufftime:,:,:,:]
        parAllTrans=torch.zeros_like(parAll)

        ## scale the parameters to real values from [0,1]
        hbvscaLst=[[1,6],[50,1000],[0.05,0.9],[0.01,0.5],[0.001,0.2],[0.2,1],
                   [0,10],[0,100],[-2.5,2.5],[0.5,10],[0,0.1],[0,0.2],[0.3,5],[0,1]]  # HBV para
        routscaLst=[[0,2.9],[0,6.5],[0,1]]  # routing para

        for ip in range(len(hbvscaLst)):  # not include routing. Scaling the parameters
            parAllTrans[:,:,ip,:]=hbvscaLst[ip][0]+parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        Nstep,Ngrid=P.size()

        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        parstaFull=parAllTrans[staind,:,:,:].unsqueeze(0).repeat([Nstep,1,1,1])  # static para matrix
        parhbvFull=torch.clone(parstaFull)
        # create probability mask for each parameter on the basin dimension
        pmat=torch.ones([1,Ngrid,1])*dydrop
        for ix in tdlst:
            staPar=parstaFull[:,:,ix-1,:]
            dynPar=parAllTrans[:,:,ix-1,:]
            drmask=torch.bernoulli(pmat).detach_().to(parhbvFull)  # to drop dynamic parameters as static in some basins
            comPar=dynPar*(1-drmask)+staPar*drmask
            parhbvFull[:,:,ix-1,:]=comPar

        # Initialize time series of model variables to save results
        Qsimmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        ETmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        SWEmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        # Output the box components of Q
        Qsimmu0=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu1=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu2=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)

        # # Not used. Logging the state variables for debug.
        # # SMlog = np.zeros(P.size())
        # logSM = np.zeros(P.size())
        # logPS = np.zeros(P.size())
        # logswet = np.zeros(P.size())
        # logRE = np.zeros(P.size())

        for t in range(Nstep):
            with torch.no_grad() if t<bufftime_old else torch.enable_grad():
                paraLst=[]

                for ip in range(len(hbvscaLst)):  # unpack HBV parameters
                    paraLst.append(parhbvFull[t,:,ip,:])

                parBETA,parFC,parK0,parK1,parK2,parLP,parPERC,parUZL,parTT,parCFMAX,parCFR,parCWH,parBETAET,parC=paraLst
                # Separate precipitation into liquid and solid components
                PRECIP=Pm[t,:,:]
                TEMP=Tm[t,:,:]
                EVAP_P=ETpm[t,:,:]

                RAIN=torch.mul(PRECIP,(TEMP>=parTT).type(torch.float32))
                SNOW=torch.mul(PRECIP,(TEMP<parTT).type(torch.float32))

                # Snow process
                SNOWPACK=SNOWPACK+SNOW
                melt=parCFMAX*(TEMP-parTT)
                melt=torch.clamp(melt,min=0.0)
                melt=torch.min(melt,SNOWPACK)
                MELTWATER=MELTWATER+melt
                SNOWPACK=SNOWPACK-melt
                refreezing=parCFR*parCFMAX*(parTT-TEMP)
                refreezing=torch.clamp(refreezing,min=0.0)
                refreezing=torch.min(refreezing,MELTWATER)
                SNOWPACK=SNOWPACK+refreezing
                MELTWATER=MELTWATER-refreezing
                tosoil=MELTWATER-(parCWH*SNOWPACK)
                tosoil=torch.clamp(tosoil,min=0.0)
                MELTWATER=MELTWATER-tosoil

                # Soil and evaporation
                soil_wetness=(SM/parFC)**parBETA
                # soil_wetness=self.HSC(zd_feature,SM/parFC)
                # soil_wetness=batch_poly_eval(SM/parFC,zd_feature[:,:,:])
                # HSC_input_ori = torch.cat(((SM/parFC)[:,None],zd_feature),dim=1)
                # batch_size,height,width=HSC_input_ori.shape
                # HSC_input = HSC_input_ori.view(batch_size, height*width)
                # soil_wetness = torch.sigmoid(self.HSC(HSC_input))
                soil_wetness=torch.clamp(soil_wetness,min=0.0,max=1.0)
                recharge=(RAIN+tosoil)*soil_wetness

                # Not used, logging states for checking
                # logSM[t,:] = SM.detach().cpu().numpy()
                # logPS[t,:] = (RAIN + tosoil).detach().cpu().numpy()
                # logswet[t,:] = (SM / parFC).detach().cpu().numpy()
                # logRE[t, :] = recharge.detach().cpu().numpy()

                SM=SM+RAIN+tosoil-recharge
                excess=SM-parFC
                excess=torch.clamp(excess,min=0.0)
                SM=SM-excess
                # MODIFY here. Different as class HBVMulT(). Add a ET shape parameter parBETAET
                evapfactor=(SM/(parLP*parFC))**parBETAET
                evapfactor=torch.clamp(evapfactor,min=0.0,max=1.0)
                ETact=EVAP_P*evapfactor
                ETact=torch.min(SM,ETact)
                SM=torch.clamp(SM-ETact,min=PRECS)  # SM can not be zero for gradient tracking

                capillary=torch.min(SLZ,parC*SLZ*(1.0-torch.clamp(SM/parFC,max=1.0)))

                SM=torch.clamp(SM+capillary,min=PRECS)
                SLZ=torch.clamp(SLZ-capillary,min=PRECS)

                # Groundwater boxes
                SUZ=SUZ+recharge+excess
                PERC=torch.min(SUZ,parPERC)
                SUZ=SUZ-PERC
                Q0=parK0*torch.clamp(SUZ-parUZL,min=0.0)
                SUZ=SUZ-Q0
                Q1=parK1*SUZ
                SUZ=SUZ-Q1
                SLZ=SLZ+PERC
                Q2=parK2*SLZ
                SLZ=SLZ-Q2
                Qsimmu[t,:,:]=Q0+Q1+Q2

                # save components for Q
                Qsimmu0[t,:,:]=Q0
                Qsimmu1[t,:,:]=Q1
                Qsimmu2[t,:,:]=Q2
                ETmu[t,:,:]=ETact
                SWEmu[t,:,:]=SNOWPACK
                # Not used, for debug state variables
                # SMlog[t,:, :] = SM.detach().cpu().numpy()
                # SUZlog[t,:,:] = SUZ.detach().cpu().numpy()
                # SLZlog[t,:,:] = SLZ.detach().cpu().numpy()

        Qsimave0=Qsimmu0.mean(-1,keepdim=True)
        Qsimave1=Qsimmu1.mean(-1,keepdim=True)
        Qsimave2=Qsimmu2.mean(-1,keepdim=True)
        ETave=ETmu.mean(-1,keepdim=True)
        SWEave=SWEmu.mean(-1,keepdim=True)

        # get the initial average
        if muwts is None:  # simple average
            Qsimave=Qsimmu.mean(-1)
        else:  # weighted average using learned weights
            Qsimave=(Qsimmu*muwts).sum(-1)

        if routOpt is True:  # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim=Qsimmu.view(Nstep,Ngrid*mu)
            else:
                # average the components, then do routing
                Qsim=Qsimave

            # scale two routing parameters
            tempa=routscaLst[0][0]+rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb=routscaLst[1][0]+rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])
            tempc=routscaLst[2][0]+rtwts[:,2]*(routscaLst[2][1]-routscaLst[2][0])
            routa=tempa.repeat(Nstep,1).unsqueeze(-1)
            routb=tempb.repeat(Nstep,1).unsqueeze(-1)
            routc=tempc.repeat(Nstep,1).unsqueeze(-1)
            UH=UH_gamma(routa,routb,lenF=15)  # lenF: folter
            rf=torch.unsqueeze(Qsim,-1).permute([1,2,0])  # dim:gage*var*time
            UH=UH.permute([1,2,0])  # dim: gage*var*time
            # Qsrout=UH_conv(rf,UH).permute([2,0,1])-wateruse[bufftime:]*routc
            # pjli relu function
            Qsrout=torch.clamp(UH_conv(rf,UH).permute([2,0,1])-wateruse[bufftime:]*routc,min=0.0)

            if comprout is True:  # Qs is [time, [gage*mult], var] now
                Qstemp=Qsrout.view(Nstep,Ngrid,mu)
                if muwts is None:
                    Qs=Qstemp.mean(-1,keepdim=True)
                else:
                    Qs=(Qstemp*muwts).sum(-1,keepdim=True)
            else:
                Qs=Qsrout

        else:  # no routing, output the initial average simulations
            Qs=torch.unsqueeze(Qsimave,-1)  # add a dimension

        if outstate is True:  # output states
            return Qs,SNOWPACK,MELTWATER,SM,SUZ,SLZ
        else:
            # return Qs
            Qall=torch.cat((Qs,Qsimave0,Qsimave1,Qsimave2,ETave,SWEave),dim=-1)
            return Qall[bufftime_old:]

class HBVMulTDET_1_1p_wateruse_loss(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""

    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first
    # 6 coefficiences are estimated by NN
    # HSC is estimated by NN

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMulTDET_1_1p_wateruse_loss,self).__init__()


    def forward(self,x,parameters, wateruse, staind,tdlst,mu,muwts,rtwts,bufftime=0,outstate=False,routOpt=False,
                comprout=False,dydrop=False):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     X: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2]
        #     staind:use which time step from the learned para time series for static parameters
        #     tdlst: the index list of hbv parameters set as dynamic
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     dydrop: the possibility to drop a dynamic para to static to reduce potential overfitting
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET

        PRECS=1e-5  # keep the numerical calculation stable
        bufftime_old = bufftime
        bufftime=0 # pjli 20240926, donot understand why, comment it, set to dynamic warm-up
        # # Initialization to warm-up states
        # if bufftime>0:
        #     with torch.no_grad():
        #         xinit=x[0:bufftime,:,:]
        #         initmodel=HBVMulTDET_1_1p_wateruse_loss()
        #         # buffpara=parameters[bufftime-1,:,:,:] # pjli 20240926 original
        #         buffpara = parameters[0:bufftime,:,:] # pql5336, dynamic parameter warm up
        #         Qsinit,SNOWPACK,MELTWATER,SM,SUZ,SLZ=initmodel(xinit,buffpara,wateruse[0:bufftime],staind,tdlst,mu,muwts,rtwts,
        #                                                        bufftime=0,outstate=True,routOpt=False,comprout=False,
        #                                                        dydrop=False)
        # else:

        # Without warm-up bufftime=0, initialize state variables with zeros
        Ngrid=x.shape[1]
        SNOWPACK=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        MELTWATER=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SM=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SUZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SLZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(x)

        P=x[bufftime:,:,0]
        Pm=P.unsqueeze(2).repeat(1,1,mu)  # precip
        T=x[bufftime:,:,1]
        Tm=T.unsqueeze(2).repeat(1,1,mu)  # temperature
        ETpot=x[bufftime:,:,2]
        ETpm=ETpot.unsqueeze(2).repeat(1,1,mu)  # potential ET
        parAll=parameters[bufftime:,:,:,:]
        parAllTrans=torch.zeros_like(parAll)

        ## scale the parameters to real values from [0,1]
        hbvscaLst=[[1,6],[50,1000],[0.05,0.9],[0.01,0.5],[0.001,0.2],[0.2,1],
                   [0,10],[0,100],[-2.5,2.5],[0.5,10],[0,0.1],[0,0.2],[0.3,5],[0,1]]  # HBV para
        routscaLst=[[0,2.9],[0,6.5],[0,1]]  # routing para

        for ip in range(len(hbvscaLst)):  # not include routing. Scaling the parameters
            parAllTrans[:,:,ip,:]=hbvscaLst[ip][0]+parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        Nstep,Ngrid=P.size()

        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        parstaFull=parAllTrans[staind,:,:,:].unsqueeze(0).repeat([Nstep,1,1,1])  # static para matrix
        parhbvFull=torch.clone(parstaFull)
        # create probability mask for each parameter on the basin dimension
        pmat=torch.ones([1,Ngrid,1])*dydrop
        for ix in tdlst:
            staPar=parstaFull[:,:,ix-1,:]
            dynPar=parAllTrans[:,:,ix-1,:]
            drmask=torch.bernoulli(pmat).detach_().to(parhbvFull)  # to drop dynamic parameters as static in some basins
            comPar=dynPar*(1-drmask)+staPar*drmask
            parhbvFull[:,:,ix-1,:]=comPar

        # Initialize time series of model variables to save results
        Qsimmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        ETmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        SWEmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        # Output the box components of Q
        Qsimmu0=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu1=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu2=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)

        # # Not used. Logging the state variables for debug.
        # # SMlog = np.zeros(P.size())
        # logSM = np.zeros(P.size())
        # logPS = np.zeros(P.size())
        # logswet = np.zeros(P.size())
        # logRE = np.zeros(P.size())

        for t in range(Nstep):
            with torch.no_grad() if t<bufftime_old else torch.enable_grad():
                paraLst=[]

                for ip in range(len(hbvscaLst)):  # unpack HBV parameters
                    paraLst.append(parhbvFull[t,:,ip,:])

                parBETA,parFC,parK0,parK1,parK2,parLP,parPERC,parUZL,parTT,parCFMAX,parCFR,parCWH,parBETAET,parC=paraLst
                # Separate precipitation into liquid and solid components
                PRECIP=Pm[t,:,:]
                TEMP=Tm[t,:,:]
                EVAP_P=ETpm[t,:,:]

                RAIN=torch.mul(PRECIP,(TEMP>=parTT).type(torch.float32))
                SNOW=torch.mul(PRECIP,(TEMP<parTT).type(torch.float32))

                # Snow process
                SNOWPACK=SNOWPACK+SNOW
                melt=parCFMAX*(TEMP-parTT)
                melt=torch.clamp(melt,min=0.0)
                melt=torch.min(melt,SNOWPACK)
                MELTWATER=MELTWATER+melt
                SNOWPACK=SNOWPACK-melt
                refreezing=parCFR*parCFMAX*(parTT-TEMP)
                refreezing=torch.clamp(refreezing,min=0.0)
                refreezing=torch.min(refreezing,MELTWATER)
                SNOWPACK=SNOWPACK+refreezing
                MELTWATER=MELTWATER-refreezing
                tosoil=MELTWATER-(parCWH*SNOWPACK)
                tosoil=torch.clamp(tosoil,min=0.0)
                MELTWATER=MELTWATER-tosoil

                # Soil and evaporation
                soil_wetness=(SM/parFC)**parBETA
                # soil_wetness=self.HSC(zd_feature,SM/parFC)
                # soil_wetness=batch_poly_eval(SM/parFC,zd_feature[:,:,:])
                # HSC_input_ori = torch.cat(((SM/parFC)[:,None],zd_feature),dim=1)
                # batch_size,height,width=HSC_input_ori.shape
                # HSC_input = HSC_input_ori.view(batch_size, height*width)
                # soil_wetness = torch.sigmoid(self.HSC(HSC_input))
                soil_wetness=torch.clamp(soil_wetness,min=0.0,max=1.0)
                recharge=(RAIN+tosoil)*soil_wetness

                # Not used, logging states for checking
                # logSM[t,:] = SM.detach().cpu().numpy()
                # logPS[t,:] = (RAIN + tosoil).detach().cpu().numpy()
                # logswet[t,:] = (SM / parFC).detach().cpu().numpy()
                # logRE[t, :] = recharge.detach().cpu().numpy()

                SM=SM+RAIN+tosoil-recharge
                excess=SM-parFC
                excess=torch.clamp(excess,min=0.0)
                SM=SM-excess
                # MODIFY here. Different as class HBVMulT(). Add a ET shape parameter parBETAET
                evapfactor=(SM/(parLP*parFC))**parBETAET
                evapfactor=torch.clamp(evapfactor,min=0.0,max=1.0)
                ETact=EVAP_P*evapfactor
                ETact=torch.min(SM,ETact)
                SM=torch.clamp(SM-ETact,min=PRECS)  # SM can not be zero for gradient tracking

                capillary=torch.min(SLZ,parC*SLZ*(1.0-torch.clamp(SM/parFC,max=1.0)))

                SM=torch.clamp(SM+capillary,min=PRECS)
                SLZ=torch.clamp(SLZ-capillary,min=PRECS)

                # Groundwater boxes
                SUZ=SUZ+recharge+excess
                PERC=torch.min(SUZ,parPERC)
                SUZ=SUZ-PERC
                Q0=parK0*torch.clamp(SUZ-parUZL,min=0.0)
                SUZ=SUZ-Q0
                Q1=parK1*SUZ
                SUZ=SUZ-Q1
                SLZ=SLZ+PERC
                Q2=parK2*SLZ
                SLZ=SLZ-Q2
                Qsimmu[t,:,:]=Q0+Q1+Q2

                # save components for Q
                Qsimmu0[t,:,:]=Q0
                Qsimmu1[t,:,:]=Q1
                Qsimmu2[t,:,:]=Q2
                ETmu[t,:,:]=ETact
                SWEmu[t,:,:]=SNOWPACK
                # Not used, for debug state variables
                # SMlog[t,:, :] = SM.detach().cpu().numpy()
                # SUZlog[t,:,:] = SUZ.detach().cpu().numpy()
                # SLZlog[t,:,:] = SLZ.detach().cpu().numpy()

        Qsimave0=Qsimmu0.mean(-1,keepdim=True)
        Qsimave1=Qsimmu1.mean(-1,keepdim=True)
        Qsimave2=Qsimmu2.mean(-1,keepdim=True)
        ETave=ETmu.mean(-1,keepdim=True)
        SWEave=SWEmu.mean(-1,keepdim=True)

        # get the initial average
        if muwts is None:  # simple average
            Qsimave=Qsimmu.mean(-1)
        else:  # weighted average using learned weights
            Qsimave=(Qsimmu*muwts).sum(-1)

        if routOpt is True:  # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim=Qsimmu.view(Nstep,Ngrid*mu)
            else:
                # average the components, then do routing
                Qsim=Qsimave

            # scale two routing parameters
            tempa=routscaLst[0][0]+rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb=routscaLst[1][0]+rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])
            tempc=routscaLst[2][0]+rtwts[:,2]*(routscaLst[2][1]-routscaLst[2][0])
            routa=tempa.repeat(Nstep,1).unsqueeze(-1)
            routb=tempb.repeat(Nstep,1).unsqueeze(-1)
            routc=tempc.repeat(Nstep,1).unsqueeze(-1)
            UH=UH_gamma(routa,routb,lenF=15)  # lenF: folter
            rf=torch.unsqueeze(Qsim,-1).permute([1,2,0])  # dim:gage*var*time
            UH=UH.permute([1,2,0])  # dim: gage*var*time
            # Qsrout=UH_conv(rf,UH).permute([2,0,1])-wateruse[bufftime:]*routc
            # pjli relu function
            Qsrout=torch.clamp(UH_conv(rf,UH).permute([2,0,1])-wateruse[bufftime:]*routc,min=0.0)

            water_loss = UH_conv(rf,UH).permute([2,0,1]) - Qsrout

            if comprout is True:  # Qs is [time, [gage*mult], var] now
                Qstemp=Qsrout.view(Nstep,Ngrid,mu)
                if muwts is None:
                    Qs=Qstemp.mean(-1,keepdim=True)
                else:
                    Qs=(Qstemp*muwts).sum(-1,keepdim=True)
            else:
                Qs=Qsrout
        else:  # no routing, output the initial average simulations
            Qs=torch.unsqueeze(Qsimave,-1)  # add a dimension

        if outstate is True:  # output states
            return Qs,SNOWPACK,MELTWATER,SM,SUZ,SLZ
        else:
            # return Qs
            Qall=torch.cat((water_loss,routc,Qsimave1,Qsimave2,ETave,SWEave),dim=-1)
            return Qall[bufftime_old:]

class HBVMulTDET_1_1p_wateruse_u2(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""

    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first
    # not only consider Q bust also soil water, upper and lower groundwater
    # all these coefficient sum up 1

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMulTDET_1_1p_wateruse_u2,self).__init__()


    def forward(self,x,parameters, wateruse, staind,tdlst,mu,muwts,rtwts,bufftime=0,outstate=False,routOpt=False,
                comprout=False,dydrop=False):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     X: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2]
        #     staind:use which time step from the learned para time series for static parameters
        #     tdlst: the index list of hbv parameters set as dynamic
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     dydrop: the possibility to drop a dynamic para to static to reduce potential overfitting
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET

        PRECS=1e-5  # keep the numerical calculation stable
        bufftime_old = bufftime
        bufftime=0 # pjli 20240926, donot understand why, comment it, set to dynamic warm-up
        # # Initialization to warm-up states
        # if bufftime>0:
        #     with torch.no_grad():
        #         xinit=x[0:bufftime,:,:]
        #         initmodel=HBVMulTDET_1_1p_wateruse_loss()
        #         # buffpara=parameters[bufftime-1,:,:,:] # pjli 20240926 original
        #         buffpara = parameters[0:bufftime,:,:] # pql5336, dynamic parameter warm up
        #         Qsinit,SNOWPACK,MELTWATER,SM,SUZ,SLZ=initmodel(xinit,buffpara,wateruse[0:bufftime],staind,tdlst,mu,muwts,rtwts,
        #                                                        bufftime=0,outstate=True,routOpt=False,comprout=False,
        #                                                        dydrop=False)
        # else:

        # Without warm-up bufftime=0, initialize state variables with zeros
        Ngrid=x.shape[1]
        SNOWPACK=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        MELTWATER=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SM=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SUZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SLZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(x)

        P=x[bufftime:,:,0]
        Pm=P.unsqueeze(2).repeat(1,1,mu)  # precip
        T=x[bufftime:,:,1]
        Tm=T.unsqueeze(2).repeat(1,1,mu)  # temperature
        ETpot=x[bufftime:,:,2]
        ETpm=ETpot.unsqueeze(2).repeat(1,1,mu)  # potential ET
        parAll=parameters[bufftime:,:,:,:]
        parAllTrans=torch.zeros_like(parAll)

        ## scale the parameters to real values from [0,1]
        hbvscaLst=[[1,6],[50,1000],[0.05,0.9],[0.01,0.5],[0.001,0.2],[0.2,1],
                   [0,10],[0,100],[-2.5,2.5],[0.5,10],[0,0.1],[0,0.2],[0.3,5],[0,1]]  # HBV para
        routscaLst=[[0,2.9],[0,6.5],[0.00001,1],[0.00001,1],[0.00001,1],[0.00001,1]]  # routing para



        for ip in range(len(hbvscaLst)):  # not include routing. Scaling the parameters
            parAllTrans[:,:,ip,:]=hbvscaLst[ip][0]+parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        Nstep,Ngrid=P.size()
        # water use parameters
        tempc1=routscaLst[2][0]+rtwts[:,2]*(routscaLst[2][1]-routscaLst[2][0])
        tempc2=routscaLst[3][0]+rtwts[:,3]*(routscaLst[3][1]-routscaLst[3][0])
        tempc3=routscaLst[4][0]+rtwts[:,4]*(routscaLst[4][1]-routscaLst[4][0])
        tempc4=routscaLst[5][0]+rtwts[:,5]*(routscaLst[5][1]-routscaLst[5][0])
        tempc_all=tempc1+tempc2+tempc3+tempc4
        routc1=(tempc1/(tempc_all)).repeat(Nstep,1).unsqueeze(-1)
        routc2=(tempc2/(tempc_all)).repeat(Nstep,1).unsqueeze(-1)
        routc3=(tempc3/(tempc_all)).repeat(Nstep,1).unsqueeze(-1)
        routc4=(tempc4/(tempc_all)).repeat(Nstep,1).unsqueeze(-1)


        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        parstaFull=parAllTrans[staind,:,:,:].unsqueeze(0).repeat([Nstep,1,1,1])  # static para matrix
        parhbvFull=torch.clone(parstaFull)
        # create probability mask for each parameter on the basin dimension
        pmat=torch.ones([1,Ngrid,1])*dydrop
        for ix in tdlst:
            staPar=parstaFull[:,:,ix-1,:]
            dynPar=parAllTrans[:,:,ix-1,:]
            drmask=torch.bernoulli(pmat).detach_().to(parhbvFull)  # to drop dynamic parameters as static in some basins
            comPar=dynPar*(1-drmask)+staPar*drmask
            parhbvFull[:,:,ix-1,:]=comPar

        # Initialize time series of model variables to save results
        Qsimmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        ETmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        SWEmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        # Output the box components of Q
        Qsimmu0=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu1=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu2=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)

        # # Not used. Logging the state variables for debug.
        # # SMlog = np.zeros(P.size())
        # logSM = np.zeros(P.size())
        # logPS = np.zeros(P.size())
        # logswet = np.zeros(P.size())
        # logRE = np.zeros(P.size())

        for t in range(Nstep):
            with torch.no_grad() if t<bufftime_old else torch.enable_grad():
                paraLst=[]

                for ip in range(len(hbvscaLst)):  # unpack HBV parameters
                    paraLst.append(parhbvFull[t,:,ip,:])

                parBETA,parFC,parK0,parK1,parK2,parLP,parPERC,parUZL,parTT,parCFMAX,parCFR,parCWH,parBETAET,parC=paraLst
                # Separate precipitation into liquid and solid components
                PRECIP=Pm[t,:,:]
                TEMP=Tm[t,:,:]
                EVAP_P=ETpm[t,:,:]

                RAIN=torch.mul(PRECIP,(TEMP>=parTT).type(torch.float32))
                SNOW=torch.mul(PRECIP,(TEMP<parTT).type(torch.float32))

                # Snow process
                SNOWPACK=SNOWPACK+SNOW
                melt=parCFMAX*(TEMP-parTT)
                melt=torch.clamp(melt,min=0.0)
                melt=torch.min(melt,SNOWPACK)
                MELTWATER=MELTWATER+melt
                SNOWPACK=SNOWPACK-melt
                refreezing=parCFR*parCFMAX*(parTT-TEMP)
                refreezing=torch.clamp(refreezing,min=0.0)
                refreezing=torch.min(refreezing,MELTWATER)
                SNOWPACK=SNOWPACK+refreezing
                MELTWATER=MELTWATER-refreezing
                tosoil=MELTWATER-(parCWH*SNOWPACK)
                tosoil=torch.clamp(tosoil,min=0.0)
                MELTWATER=MELTWATER-tosoil

                # Soil and evaporation
                soil_wetness=(SM/parFC)**parBETA
                # soil_wetness=self.HSC(zd_feature,SM/parFC)
                # soil_wetness=batch_poly_eval(SM/parFC,zd_feature[:,:,:])
                # HSC_input_ori = torch.cat(((SM/parFC)[:,None],zd_feature),dim=1)
                # batch_size,height,width=HSC_input_ori.shape
                # HSC_input = HSC_input_ori.view(batch_size, height*width)
                # soil_wetness = torch.sigmoid(self.HSC(HSC_input))
                soil_wetness=torch.clamp(soil_wetness,min=0.0,max=1.0)
                recharge=(RAIN+tosoil)*soil_wetness

                # Not used, logging states for checking
                # logSM[t,:] = SM.detach().cpu().numpy()
                # logPS[t,:] = (RAIN + tosoil).detach().cpu().numpy()
                # logswet[t,:] = (SM / parFC).detach().cpu().numpy()
                # logRE[t, :] = recharge.detach().cpu().numpy()

                SM=SM+RAIN+tosoil-recharge
                excess=SM-parFC
                excess=torch.clamp(excess,min=0.0)
                SM=SM-excess
                # MODIFY here. Different as class HBVMulT(). Add a ET shape parameter parBETAET
                evapfactor=(SM/(parLP*parFC))**parBETAET
                evapfactor=torch.clamp(evapfactor,min=0.0,max=1.0)
                ETact=EVAP_P*evapfactor
                ETact=torch.min(SM,ETact)
                SM=torch.clamp(SM-ETact,min=PRECS)  # SM can not be zero for gradient tracking

                capillary=torch.min(SLZ,parC*SLZ*(1.0-torch.clamp(SM/parFC,max=1.0)))

                SM=torch.clamp(SM+capillary,min=PRECS)
                SLZ=torch.clamp(SLZ-capillary,min=PRECS)

                # Groundwater boxes
                SUZ=SUZ+recharge+excess
                PERC=torch.min(SUZ,parPERC)
                SUZ=SUZ-PERC
                Q0=parK0*torch.clamp(SUZ-parUZL,min=0.0)
                SUZ=SUZ-Q0
                Q1=parK1*SUZ
                SUZ=SUZ-Q1
                SLZ=SLZ+PERC
                Q2=parK2*SLZ
                SLZ=SLZ-Q2
                Qsimmu[t,:,:]=Q0+Q1+Q2

                # save components for Q
                Qsimmu0[t,:,:]=Q0
                Qsimmu1[t,:,:]=Q1
                Qsimmu2[t,:,:]=Q2
                ETmu[t,:,:]=ETact
                SWEmu[t,:,:]=SNOWPACK
                # Not used, for debug state variables
                # SMlog[t,:, :] = SM.detach().cpu().numpy()
                # SUZlog[t,:,:] = SUZ.detach().cpu().numpy()
                # SLZlog[t,:,:] = SLZ.detach().cpu().numpy()

                SM=torch.clamp(SM-wateruse[bufftime:][t]*routc2[t],min=PRECS)
                SUZ=torch.clamp(SUZ-wateruse[bufftime:][t]*routc3[t],min=PRECS)
                SLZ=torch.clamp(SLZ-wateruse[bufftime:][t]*routc4[t],min=PRECS)


        Qsimave0=Qsimmu0.mean(-1,keepdim=True)
        Qsimave1=Qsimmu1.mean(-1,keepdim=True)
        Qsimave2=Qsimmu2.mean(-1,keepdim=True)
        ETave=ETmu.mean(-1,keepdim=True)
        SWEave=SWEmu.mean(-1,keepdim=True)

        # get the initial average
        if muwts is None:  # simple average
            Qsimave=Qsimmu.mean(-1)
        else:  # weighted average using learned weights
            Qsimave=(Qsimmu*muwts).sum(-1)

        if routOpt is True:  # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim=Qsimmu.view(Nstep,Ngrid*mu)
            else:
                # average the components, then do routing
                Qsim=Qsimave

            # scale two routing parameters
            tempa=routscaLst[0][0]+rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb=routscaLst[1][0]+rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])

            routa=tempa.repeat(Nstep,1).unsqueeze(-1)
            routb=tempb.repeat(Nstep,1).unsqueeze(-1)

            UH=UH_gamma(routa,routb,lenF=15)  # lenF: folter
            rf=torch.unsqueeze(Qsim,-1).permute([1,2,0])  # dim:gage*var*time
            UH=UH.permute([1,2,0])  # dim: gage*var*time
            # Qsrout=UH_conv(rf,UH).permute([2,0,1])-wateruse[bufftime:]*routc
            # pjli relu function
            Qsrout=torch.clamp(UH_conv(rf,UH).permute([2,0,1])-wateruse[bufftime:]*routc1,min=0.0)

            if comprout is True:  # Qs is [time, [gage*mult], var] now
                Qstemp=Qsrout.view(Nstep,Ngrid,mu)
                if muwts is None:
                    Qs=Qstemp.mean(-1,keepdim=True)
                else:
                    Qs=(Qstemp*muwts).sum(-1,keepdim=True)
            else:
                Qs=Qsrout

        else:  # no routing, output the initial average simulations
            Qs=torch.unsqueeze(Qsimave,-1)  # add a dimension

        if outstate is True:  # output states
            return Qs,SNOWPACK,MELTWATER,SM,SUZ,SLZ
        else:
            # return Qs
            Qall=torch.cat((Qs,Qsimave0,Qsimave1,Qsimave2,ETave,SWEave),dim=-1)
            return Qall[bufftime_old:]



class HBVMulTDET_1_1p_wateruse_u2_loss(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""

    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first
    # not only consider Q bust also soil water, upper and lower groundwater
    # all these coefficient sum up 1

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMulTDET_1_1p_wateruse_u2_loss,self).__init__()


    def forward(self,x,parameters, wateruse, staind,tdlst,mu,muwts,rtwts,bufftime=0,outstate=False,routOpt=False,
                comprout=False,dydrop=False):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     X: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2]
        #     staind:use which time step from the learned para time series for static parameters
        #     tdlst: the index list of hbv parameters set as dynamic
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     dydrop: the possibility to drop a dynamic para to static to reduce potential overfitting
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET

        PRECS=1e-5  # keep the numerical calculation stable
        bufftime_old = bufftime
        bufftime=0 # pjli 20240926, donot understand why, comment it, set to dynamic warm-up
        # # Initialization to warm-up states
        # if bufftime>0:
        #     with torch.no_grad():
        #         xinit=x[0:bufftime,:,:]
        #         initmodel=HBVMulTDET_1_1p_wateruse_loss()
        #         # buffpara=parameters[bufftime-1,:,:,:] # pjli 20240926 original
        #         buffpara = parameters[0:bufftime,:,:] # pql5336, dynamic parameter warm up
        #         Qsinit,SNOWPACK,MELTWATER,SM,SUZ,SLZ=initmodel(xinit,buffpara,wateruse[0:bufftime],staind,tdlst,mu,muwts,rtwts,
        #                                                        bufftime=0,outstate=True,routOpt=False,comprout=False,
        #                                                        dydrop=False)
        # else:

        # Without warm-up bufftime=0, initialize state variables with zeros
        Ngrid=x.shape[1]
        SNOWPACK=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        MELTWATER=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SM=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SUZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SLZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(x)

        P=x[bufftime:,:,0]
        Pm=P.unsqueeze(2).repeat(1,1,mu)  # precip
        T=x[bufftime:,:,1]
        Tm=T.unsqueeze(2).repeat(1,1,mu)  # temperature
        ETpot=x[bufftime:,:,2]
        ETpm=ETpot.unsqueeze(2).repeat(1,1,mu)  # potential ET
        parAll=parameters[bufftime:,:,:,:]
        parAllTrans=torch.zeros_like(parAll)

        ## scale the parameters to real values from [0,1]
        hbvscaLst=[[1,6],[50,1000],[0.05,0.9],[0.01,0.5],[0.001,0.2],[0.2,1],
                   [0,10],[0,100],[-2.5,2.5],[0.5,10],[0,0.1],[0,0.2],[0.3,5],[0,1]]  # HBV para
        routscaLst=[[0,2.9],[0,6.5],[0.00001,1],[0.00001,1],[0.00001,1],[0.00001,1]]  # routing para



        for ip in range(len(hbvscaLst)):  # not include routing. Scaling the parameters
            parAllTrans[:,:,ip,:]=hbvscaLst[ip][0]+parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        Nstep,Ngrid=P.size()
        # water use parameters
        tempc1=routscaLst[2][0]+rtwts[:,2]*(routscaLst[2][1]-routscaLst[2][0])
        tempc2=routscaLst[3][0]+rtwts[:,3]*(routscaLst[3][1]-routscaLst[3][0])
        tempc3=routscaLst[4][0]+rtwts[:,4]*(routscaLst[4][1]-routscaLst[4][0])
        tempc4=routscaLst[5][0]+rtwts[:,5]*(routscaLst[5][1]-routscaLst[5][0])
        tempc_all=tempc1+tempc2+tempc3+tempc4
        routc1=(tempc1/(tempc_all)).repeat(Nstep,1).unsqueeze(-1)
        routc2=(tempc2/(tempc_all)).repeat(Nstep,1).unsqueeze(-1)
        routc3=(tempc3/(tempc_all)).repeat(Nstep,1).unsqueeze(-1)
        routc4=(tempc4/(tempc_all)).repeat(Nstep,1).unsqueeze(-1)


        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        parstaFull=parAllTrans[staind,:,:,:].unsqueeze(0).repeat([Nstep,1,1,1])  # static para matrix
        parhbvFull=torch.clone(parstaFull)
        # create probability mask for each parameter on the basin dimension
        pmat=torch.ones([1,Ngrid,1])*dydrop
        for ix in tdlst:
            staPar=parstaFull[:,:,ix-1,:]
            dynPar=parAllTrans[:,:,ix-1,:]
            drmask=torch.bernoulli(pmat).detach_().to(parhbvFull)  # to drop dynamic parameters as static in some basins
            comPar=dynPar*(1-drmask)+staPar*drmask
            parhbvFull[:,:,ix-1,:]=comPar

        # Initialize time series of model variables to save results
        Qsimmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        ETmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        SWEmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        # Output the box components of Q
        Qsimmu0=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu1=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu2=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)

        # # Not used. Logging the state variables for debug.
        # # SMlog = np.zeros(P.size())
        # logSM = np.zeros(P.size())
        # logPS = np.zeros(P.size())
        # logswet = np.zeros(P.size())
        # logRE = np.zeros(P.size())

        for t in range(Nstep):
            with torch.no_grad() if t<bufftime_old else torch.enable_grad():
                paraLst=[]

                for ip in range(len(hbvscaLst)):  # unpack HBV parameters
                    paraLst.append(parhbvFull[t,:,ip,:])

                parBETA,parFC,parK0,parK1,parK2,parLP,parPERC,parUZL,parTT,parCFMAX,parCFR,parCWH,parBETAET,parC=paraLst
                # Separate precipitation into liquid and solid components
                PRECIP=Pm[t,:,:]
                TEMP=Tm[t,:,:]
                EVAP_P=ETpm[t,:,:]

                RAIN=torch.mul(PRECIP,(TEMP>=parTT).type(torch.float32))
                SNOW=torch.mul(PRECIP,(TEMP<parTT).type(torch.float32))

                # Snow process
                SNOWPACK=SNOWPACK+SNOW
                melt=parCFMAX*(TEMP-parTT)
                melt=torch.clamp(melt,min=0.0)
                melt=torch.min(melt,SNOWPACK)
                MELTWATER=MELTWATER+melt
                SNOWPACK=SNOWPACK-melt
                refreezing=parCFR*parCFMAX*(parTT-TEMP)
                refreezing=torch.clamp(refreezing,min=0.0)
                refreezing=torch.min(refreezing,MELTWATER)
                SNOWPACK=SNOWPACK+refreezing
                MELTWATER=MELTWATER-refreezing
                tosoil=MELTWATER-(parCWH*SNOWPACK)
                tosoil=torch.clamp(tosoil,min=0.0)
                MELTWATER=MELTWATER-tosoil

                # Soil and evaporation
                soil_wetness=(SM/parFC)**parBETA
                # soil_wetness=self.HSC(zd_feature,SM/parFC)
                # soil_wetness=batch_poly_eval(SM/parFC,zd_feature[:,:,:])
                # HSC_input_ori = torch.cat(((SM/parFC)[:,None],zd_feature),dim=1)
                # batch_size,height,width=HSC_input_ori.shape
                # HSC_input = HSC_input_ori.view(batch_size, height*width)
                # soil_wetness = torch.sigmoid(self.HSC(HSC_input))
                soil_wetness=torch.clamp(soil_wetness,min=0.0,max=1.0)
                recharge=(RAIN+tosoil)*soil_wetness

                # Not used, logging states for checking
                # logSM[t,:] = SM.detach().cpu().numpy()
                # logPS[t,:] = (RAIN + tosoil).detach().cpu().numpy()
                # logswet[t,:] = (SM / parFC).detach().cpu().numpy()
                # logRE[t, :] = recharge.detach().cpu().numpy()

                SM=SM+RAIN+tosoil-recharge
                excess=SM-parFC
                excess=torch.clamp(excess,min=0.0)
                SM=SM-excess
                # MODIFY here. Different as class HBVMulT(). Add a ET shape parameter parBETAET
                evapfactor=(SM/(parLP*parFC))**parBETAET
                evapfactor=torch.clamp(evapfactor,min=0.0,max=1.0)
                ETact=EVAP_P*evapfactor
                ETact=torch.min(SM,ETact)
                SM=torch.clamp(SM-ETact,min=PRECS)  # SM can not be zero for gradient tracking

                capillary=torch.min(SLZ,parC*SLZ*(1.0-torch.clamp(SM/parFC,max=1.0)))

                SM=torch.clamp(SM+capillary,min=PRECS)
                SLZ=torch.clamp(SLZ-capillary,min=PRECS)

                # Groundwater boxes
                SUZ=SUZ+recharge+excess
                PERC=torch.min(SUZ,parPERC)
                SUZ=SUZ-PERC
                Q0=parK0*torch.clamp(SUZ-parUZL,min=0.0)
                SUZ=SUZ-Q0
                Q1=parK1*SUZ
                SUZ=SUZ-Q1
                SLZ=SLZ+PERC
                Q2=parK2*SLZ
                SLZ=SLZ-Q2
                Qsimmu[t,:,:]=Q0+Q1+Q2

                # save components for Q
                Qsimmu0[t,:,:]=Q0
                Qsimmu1[t,:,:]=Q1
                Qsimmu2[t,:,:]=Q2
                ETmu[t,:,:]=ETact
                SWEmu[t,:,:]=SNOWPACK
                # Not used, for debug state variables
                # SMlog[t,:, :] = SM.detach().cpu().numpy()
                # SUZlog[t,:,:] = SUZ.detach().cpu().numpy()
                # SLZlog[t,:,:] = SLZ.detach().cpu().numpy()

                SM=torch.clamp(SM-wateruse[bufftime:][t]*routc2[t],min=PRECS)
                SUZ=torch.clamp(SUZ-wateruse[bufftime:][t]*routc3[t],min=PRECS)
                SLZ=torch.clamp(SLZ-wateruse[bufftime:][t]*routc4[t],min=PRECS)


        Qsimave0=Qsimmu0.mean(-1,keepdim=True)
        Qsimave1=Qsimmu1.mean(-1,keepdim=True)
        Qsimave2=Qsimmu2.mean(-1,keepdim=True)
        ETave=ETmu.mean(-1,keepdim=True)
        SWEave=SWEmu.mean(-1,keepdim=True)

        # get the initial average
        if muwts is None:  # simple average
            Qsimave=Qsimmu.mean(-1)
        else:  # weighted average using learned weights
            Qsimave=(Qsimmu*muwts).sum(-1)

        if routOpt is True:  # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim=Qsimmu.view(Nstep,Ngrid*mu)
            else:
                # average the components, then do routing
                Qsim=Qsimave

            # scale two routing parameters
            tempa=routscaLst[0][0]+rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb=routscaLst[1][0]+rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])

            routa=tempa.repeat(Nstep,1).unsqueeze(-1)
            routb=tempb.repeat(Nstep,1).unsqueeze(-1)

            UH=UH_gamma(routa,routb,lenF=15)  # lenF: folter
            rf=torch.unsqueeze(Qsim,-1).permute([1,2,0])  # dim:gage*var*time
            UH=UH.permute([1,2,0])  # dim: gage*var*time
            # Qsrout=UH_conv(rf,UH).permute([2,0,1])-wateruse[bufftime:]*routc
            # pjli relu function
            Qsrout=torch.clamp(UH_conv(rf,UH).permute([2,0,1])-wateruse[bufftime:]*routc1,min=0.0)
            water_loss=UH_conv(rf,UH).permute([2,0,1])-Qsrout

            if comprout is True:  # Qs is [time, [gage*mult], var] now
                Qstemp=Qsrout.view(Nstep,Ngrid,mu)
                if muwts is None:
                    Qs=Qstemp.mean(-1,keepdim=True)
                else:
                    Qs=(Qstemp*muwts).sum(-1,keepdim=True)
            else:
                Qs=Qsrout

        else:  # no routing, output the initial average simulations
            Qs=torch.unsqueeze(Qsimave,-1)  # add a dimension

        if outstate is True:  # output states
            return Qs,SNOWPACK,MELTWATER,SM,SUZ,SLZ
        else:
            # return Qs
            Qall=torch.cat((water_loss,routc1,routc2,routc3,routc4,SWEave),dim=-1)
            return Qall[bufftime_old:]


class HBVMulTDET_1_1p_wateruse_u3_dyn(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""

    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first
    # not only consider Q bust also soil water, upper and lower groundwater
    # all these coefficient sum up 1

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMulTDET_1_1p_wateruse_u3_dyn,self).__init__()

    def forward(self,x,parameters,wateruse,staind,tdlst,mu,muwts,rtwts,bufftime=0,outstate=False,routOpt=False,
                comprout=False,dydrop=False):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     X: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2]
        #     staind:use which time step from the learned para time series for static parameters
        #     tdlst: the index list of hbv parameters set as dynamic
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     dydrop: the possibility to drop a dynamic para to static to reduce potential overfitting
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET

        PRECS=1e-5  # keep the numerical calculation stable
        bufftime_old = bufftime
        bufftime=0 # pjli 20240926, donot understand why, comment it, set to dynamic warm-up
        # # Initialization to warm-up states
        # if bufftime>0:
        #     with torch.no_grad():
        #         xinit=x[0:bufftime,:,:]
        #         initmodel=HBVMulTDET_1_1p_wateruse_loss()
        #         # buffpara=parameters[bufftime-1,:,:,:] # pjli 20240926 original
        #         buffpara = parameters[0:bufftime,:,:] # pql5336, dynamic parameter warm up
        #         Qsinit,SNOWPACK,MELTWATER,SM,SUZ,SLZ=initmodel(xinit,buffpara,wateruse[0:bufftime],staind,tdlst,mu,muwts,rtwts,
        #                                                        bufftime=0,outstate=True,routOpt=False,comprout=False,
        #                                                        dydrop=False)
        # else:

        # Without warm-up bufftime=0, initialize state variables with zeros
        Ngrid=x.shape[1]
        SNOWPACK=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        MELTWATER=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SM=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SUZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SLZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(x)

        P=x[bufftime:,:,0]
        Pm=P.unsqueeze(2).repeat(1,1,mu)  # precip
        T=x[bufftime:,:,1]
        Tm=T.unsqueeze(2).repeat(1,1,mu)  # temperature
        ETpot=x[bufftime:,:,2]
        ETpm=ETpot.unsqueeze(2).repeat(1,1,mu)  # potential ET
        parAll=parameters[bufftime:,:,:,:]
        parAllTrans=torch.zeros_like(parAll)

        ## scale the parameters to real values from [0,1]
        hbvscaLst=[[1,6],[50,1000],[0.05,0.9],[0.01,0.5],[0.001,0.2],[0.2,1],
                   [0,10],[0,100],[-2.5,2.5],[0.5,10],[0,0.1],[0,0.2],[0.3,5],[0,1],[0.00001,1],[0.00001,1],[0.00001,1],
                   [0.00001,1]]  # HBV para
        routscaLst=[[0,2.9],[0,6.5]]  # routing para

        for ip in range(len(hbvscaLst)):  # not include routing. Scaling the parameters
            parAllTrans[:,:,ip,:]=hbvscaLst[ip][0]+parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        Nstep,Ngrid=P.size()

        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        parstaFull=parAllTrans[staind,:,:,:].unsqueeze(0).repeat([Nstep,1,1,1])  # static para matrix
        parhbvFull=torch.clone(parstaFull)
        # create probability mask for each parameter on the basin dimension
        pmat=torch.ones([1,Ngrid,1])*dydrop
        for ix in tdlst:
            staPar=parstaFull[:,:,ix-1,:]
            dynPar=parAllTrans[:,:,ix-1,:]
            drmask=torch.bernoulli(pmat).detach_().to(parhbvFull)  # to drop dynamic parameters as static in some basins
            comPar=dynPar*(1-drmask)+staPar*drmask
            parhbvFull[:,:,ix-1,:]=comPar

        # Initialize time series of model variables to save results
        Qsimmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        ETmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        SWEmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        # Output the box components of Q
        Qsimmu0=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu1=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu2=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)

        # # Not used. Logging the state variables for debug.
        # # SMlog = np.zeros(P.size())
        # logSM = np.zeros(P.size())
        # logPS = np.zeros(P.size())
        # logswet = np.zeros(P.size())
        # logRE = np.zeros(P.size())

        routc1_all=(torch.zeros(P.size(),dtype=torch.float32)+0.001).to(parhbvFull).unsqueeze(-1)
        for t in range(Nstep):
            with torch.no_grad() if t<bufftime_old else torch.enable_grad():
                paraLst=[]

                for ip in range(len(hbvscaLst)):  # unpack HBV parameters
                    paraLst.append(parhbvFull[t,:,ip,:])

                parBETA,parFC,parK0,parK1,parK2,parLP,parPERC,parUZL,parTT,parCFMAX,parCFR,parCWH,parBETAET,parC,\
                    tempc1,tempc2,tempc3,tempc4=paraLst

                # water use parameters
                tempc_all=tempc1+tempc2+tempc3+tempc4
                routc1=(tempc1/(tempc_all)).mean(-1).unsqueeze(-1)
                routc2=(tempc2/(tempc_all)).mean(-1).unsqueeze(-1)
                routc3=(tempc3/(tempc_all)).mean(-1).unsqueeze(-1)
                routc4=(tempc4/(tempc_all)).mean(-1).unsqueeze(-1)

                # Separate precipitation into liquid and solid components
                PRECIP=Pm[t,:,:]
                TEMP=Tm[t,:,:]
                EVAP_P=ETpm[t,:,:]

                RAIN=torch.mul(PRECIP,(TEMP>=parTT).type(torch.float32))
                SNOW=torch.mul(PRECIP,(TEMP<parTT).type(torch.float32))

                # Snow process
                SNOWPACK=SNOWPACK+SNOW
                melt=parCFMAX*(TEMP-parTT)
                melt=torch.clamp(melt,min=0.0)
                melt=torch.min(melt,SNOWPACK)
                MELTWATER=MELTWATER+melt
                SNOWPACK=SNOWPACK-melt
                refreezing=parCFR*parCFMAX*(parTT-TEMP)
                refreezing=torch.clamp(refreezing,min=0.0)
                refreezing=torch.min(refreezing,MELTWATER)
                SNOWPACK=SNOWPACK+refreezing
                MELTWATER=MELTWATER-refreezing
                tosoil=MELTWATER-(parCWH*SNOWPACK)
                tosoil=torch.clamp(tosoil,min=0.0)
                MELTWATER=MELTWATER-tosoil

                # Soil and evaporation
                soil_wetness=(SM/parFC)**parBETA
                # soil_wetness=self.HSC(zd_feature,SM/parFC)
                # soil_wetness=batch_poly_eval(SM/parFC,zd_feature[:,:,:])
                # HSC_input_ori = torch.cat(((SM/parFC)[:,None],zd_feature),dim=1)
                # batch_size,height,width=HSC_input_ori.shape
                # HSC_input = HSC_input_ori.view(batch_size, height*width)
                # soil_wetness = torch.sigmoid(self.HSC(HSC_input))
                soil_wetness=torch.clamp(soil_wetness,min=0.0,max=1.0)
                recharge=(RAIN+tosoil)*soil_wetness

                # Not used, logging states for checking
                # logSM[t,:] = SM.detach().cpu().numpy()
                # logPS[t,:] = (RAIN + tosoil).detach().cpu().numpy()
                # logswet[t,:] = (SM / parFC).detach().cpu().numpy()
                # logRE[t, :] = recharge.detach().cpu().numpy()

                SM=SM+RAIN+tosoil-recharge
                excess=SM-parFC
                excess=torch.clamp(excess,min=0.0)
                SM=SM-excess
                # MODIFY here. Different as class HBVMulT(). Add a ET shape parameter parBETAET
                evapfactor=(SM/(parLP*parFC))**parBETAET
                evapfactor=torch.clamp(evapfactor,min=0.0,max=1.0)
                ETact=EVAP_P*evapfactor
                ETact=torch.min(SM,ETact)
                SM=torch.clamp(SM-ETact,min=PRECS)  # SM can not be zero for gradient tracking

                capillary=torch.min(SLZ,parC*SLZ*(1.0-torch.clamp(SM/parFC,max=1.0)))

                SM=torch.clamp(SM+capillary,min=PRECS)
                SLZ=torch.clamp(SLZ-capillary,min=PRECS)

                # Groundwater boxes
                SUZ=SUZ+recharge+excess
                PERC=torch.min(SUZ,parPERC)
                SUZ=SUZ-PERC
                Q0=parK0*torch.clamp(SUZ-parUZL,min=0.0)
                SUZ=SUZ-Q0
                Q1=parK1*SUZ
                SUZ=SUZ-Q1
                SLZ=SLZ+PERC
                Q2=parK2*SLZ
                SLZ=SLZ-Q2
                Qsimmu[t,:,:]=Q0+Q1+Q2

                # save components for Q
                Qsimmu0[t,:,:]=Q0
                Qsimmu1[t,:,:]=Q1
                Qsimmu2[t,:,:]=Q2
                ETmu[t,:,:]=ETact
                SWEmu[t,:,:]=SNOWPACK
                # Not used, for debug state variables
                # SMlog[t,:, :] = SM.detach().cpu().numpy()
                # SUZlog[t,:,:] = SUZ.detach().cpu().numpy()
                # SLZlog[t,:,:] = SLZ.detach().cpu().numpy()

                SM=torch.clamp(SM-wateruse[bufftime:][t]*routc2,min=PRECS)
                SUZ=torch.clamp(SUZ-wateruse[bufftime:][t]*routc3,min=PRECS)
                SLZ=torch.clamp(SLZ-wateruse[bufftime:][t]*routc4,min=PRECS)
                routc1_all[t]=routc1

        Qsimave0=Qsimmu0.mean(-1,keepdim=True)
        Qsimave1=Qsimmu1.mean(-1,keepdim=True)
        Qsimave2=Qsimmu2.mean(-1,keepdim=True)
        ETave=ETmu.mean(-1,keepdim=True)
        SWEave=SWEmu.mean(-1,keepdim=True)

        # get the initial average
        if muwts is None:  # simple average
            Qsimave=Qsimmu.mean(-1)
        else:  # weighted average using learned weights
            Qsimave=(Qsimmu*muwts).sum(-1)

        if routOpt is True:  # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim=Qsimmu.view(Nstep,Ngrid*mu)
            else:
                # average the components, then do routing
                Qsim=Qsimave

            # scale two routing parameters
            tempa=routscaLst[0][0]+rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb=routscaLst[1][0]+rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])

            routa=tempa.repeat(Nstep,1).unsqueeze(-1)
            routb=tempb.repeat(Nstep,1).unsqueeze(-1)

            UH=UH_gamma(routa,routb,lenF=15)  # lenF: folter
            rf=torch.unsqueeze(Qsim,-1).permute([1,2,0])  # dim:gage*var*time
            UH=UH.permute([1,2,0])  # dim: gage*var*time
            # Qsrout=UH_conv(rf,UH).permute([2,0,1])-wateruse[bufftime:]*routc
            # pjli relu function
            Qsrout=torch.clamp(UH_conv(rf,UH).permute([2,0,1])-wateruse[bufftime:]*routc1_all,min=0.0)

            if comprout is True:  # Qs is [time, [gage*mult], var] now
                Qstemp=Qsrout.view(Nstep,Ngrid,mu)
                if muwts is None:
                    Qs=Qstemp.mean(-1,keepdim=True)
                else:
                    Qs=(Qstemp*muwts).sum(-1,keepdim=True)
            else:
                Qs=Qsrout

        else:  # no routing, output the initial average simulations
            Qs=torch.unsqueeze(Qsimave,-1)  # add a dimension

        if outstate is True:  # output states
            return Qs,SNOWPACK,MELTWATER,SM,SUZ,SLZ
        else:
            # return Qs
            Qall=torch.cat((Qs,Qsimave0,Qsimave1,Qsimave2,ETave,SWEave),dim=-1)
            return Qall[bufftime_old:]

class HBVMulTDET_1_1p_wateruse_u4_dyn_before(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""

    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first
    # not only consider Q bust also soil water, upper and lower groundwater
    # all these coefficient sum up 1
    # remove the consideration of soil water and abstract the water use before the implicit routing

    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMulTDET_1_1p_wateruse_u4_dyn_before,self).__init__()

    def forward(self,x,parameters,wateruse,staind,tdlst,mu,muwts,rtwts,bufftime=0,outstate=False,routOpt=False,
                comprout=False,dydrop=False):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     X: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2]
        #     staind:use which time step from the learned para time series for static parameters
        #     tdlst: the index list of hbv parameters set as dynamic
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     dydrop: the possibility to drop a dynamic para to static to reduce potential overfitting
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET

        PRECS=1e-5  # keep the numerical calculation stable
        bufftime_old = bufftime
        bufftime=0 # pjli 20240926, donot understand why, comment it, set to dynamic warm-up
        # # Initialization to warm-up states
        # if bufftime>0:
        #     with torch.no_grad():
        #         xinit=x[0:bufftime,:,:]
        #         initmodel=HBVMulTDET_1_1p_wateruse_loss()
        #         # buffpara=parameters[bufftime-1,:,:,:] # pjli 20240926 original
        #         buffpara = parameters[0:bufftime,:,:] # pql5336, dynamic parameter warm up
        #         Qsinit,SNOWPACK,MELTWATER,SM,SUZ,SLZ=initmodel(xinit,buffpara,wateruse[0:bufftime],staind,tdlst,mu,muwts,rtwts,
        #                                                        bufftime=0,outstate=True,routOpt=False,comprout=False,
        #                                                        dydrop=False)
        # else:

        # Without warm-up bufftime=0, initialize state variables with zeros
        Ngrid=x.shape[1]
        SNOWPACK=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        MELTWATER=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SM=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SUZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SLZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(x)

        P=x[bufftime:,:,0]
        Pm=P.unsqueeze(2).repeat(1,1,mu)  # precip
        T=x[bufftime:,:,1]
        Tm=T.unsqueeze(2).repeat(1,1,mu)  # temperature
        ETpot=x[bufftime:,:,2]
        ETpm=ETpot.unsqueeze(2).repeat(1,1,mu)  # potential ET
        parAll=parameters[bufftime:,:,:,:]
        parAllTrans=torch.zeros_like(parAll)

        ## scale the parameters to real values from [0,1]
        hbvscaLst=[[1,6],[50,1000],[0.05,0.9],[0.01,0.5],[0.001,0.2],[0.2,1],
                   [0,10],[0,100],[-2.5,2.5],[0.5,10],[0,0.1],[0,0.2],[0.3,5],[0,1],[0.00001,1],[0.00001,1],
                   [0.00001,1]]  # HBV para
        routscaLst=[[0,2.9],[0,6.5]]  # routing para

        for ip in range(len(hbvscaLst)):  # not include routing. Scaling the parameters
            parAllTrans[:,:,ip,:]=hbvscaLst[ip][0]+parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        Nstep,Ngrid=P.size()

        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        parstaFull=parAllTrans[staind,:,:,:].unsqueeze(0).repeat([Nstep,1,1,1])  # static para matrix
        parhbvFull=torch.clone(parstaFull)
        # create probability mask for each parameter on the basin dimension
        pmat=torch.ones([1,Ngrid,1])*dydrop
        for ix in tdlst:
            staPar=parstaFull[:,:,ix-1,:]
            dynPar=parAllTrans[:,:,ix-1,:]
            drmask=torch.bernoulli(pmat).detach_().to(parhbvFull)  # to drop dynamic parameters as static in some basins
            comPar=dynPar*(1-drmask)+staPar*drmask
            parhbvFull[:,:,ix-1,:]=comPar

        # Initialize time series of model variables to save results
        Qsimmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        ETmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        SWEmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        # Output the box components of Q
        Qsimmu0=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu1=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu2=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)

        # # Not used. Logging the state variables for debug.
        # # SMlog = np.zeros(P.size())
        # logSM = np.zeros(P.size())
        # logPS = np.zeros(P.size())
        # logswet = np.zeros(P.size())
        # logRE = np.zeros(P.size())

        routc1_all=(torch.zeros(P.size(),dtype=torch.float32)+0.001).to(parhbvFull).unsqueeze(-1)
        for t in range(Nstep):
            with torch.no_grad() if t<bufftime_old else torch.enable_grad():
                paraLst=[]

                for ip in range(len(hbvscaLst)):  # unpack HBV parameters
                    paraLst.append(parhbvFull[t,:,ip,:])

                parBETA,parFC,parK0,parK1,parK2,parLP,parPERC,parUZL,parTT,parCFMAX,parCFR,parCWH,parBETAET,parC,\
                    tempc1,tempc2,tempc3 =paraLst

                # water use parameters
                tempc_all=tempc1+tempc2+tempc3
                routc1=(tempc1/(tempc_all)).mean(-1).unsqueeze(-1)
                routc2=(tempc2/(tempc_all)).mean(-1).unsqueeze(-1)
                routc3=(tempc3/(tempc_all)).mean(-1).unsqueeze(-1)


                # Separate precipitation into liquid and solid components
                PRECIP=Pm[t,:,:]
                TEMP=Tm[t,:,:]
                EVAP_P=ETpm[t,:,:]

                RAIN=torch.mul(PRECIP,(TEMP>=parTT).type(torch.float32))
                SNOW=torch.mul(PRECIP,(TEMP<parTT).type(torch.float32))

                # Snow process
                SNOWPACK=SNOWPACK+SNOW
                melt=parCFMAX*(TEMP-parTT)
                melt=torch.clamp(melt,min=0.0)
                melt=torch.min(melt,SNOWPACK)
                MELTWATER=MELTWATER+melt
                SNOWPACK=SNOWPACK-melt
                refreezing=parCFR*parCFMAX*(parTT-TEMP)
                refreezing=torch.clamp(refreezing,min=0.0)
                refreezing=torch.min(refreezing,MELTWATER)
                SNOWPACK=SNOWPACK+refreezing
                MELTWATER=MELTWATER-refreezing
                tosoil=MELTWATER-(parCWH*SNOWPACK)
                tosoil=torch.clamp(tosoil,min=0.0)
                MELTWATER=MELTWATER-tosoil

                # Soil and evaporation
                soil_wetness=(SM/parFC)**parBETA
                # soil_wetness=self.HSC(zd_feature,SM/parFC)
                # soil_wetness=batch_poly_eval(SM/parFC,zd_feature[:,:,:])
                # HSC_input_ori = torch.cat(((SM/parFC)[:,None],zd_feature),dim=1)
                # batch_size,height,width=HSC_input_ori.shape
                # HSC_input = HSC_input_ori.view(batch_size, height*width)
                # soil_wetness = torch.sigmoid(self.HSC(HSC_input))
                soil_wetness=torch.clamp(soil_wetness,min=0.0,max=1.0)
                recharge=(RAIN+tosoil)*soil_wetness

                # Not used, logging states for checking
                # logSM[t,:] = SM.detach().cpu().numpy()
                # logPS[t,:] = (RAIN + tosoil).detach().cpu().numpy()
                # logswet[t,:] = (SM / parFC).detach().cpu().numpy()
                # logRE[t, :] = recharge.detach().cpu().numpy()

                SM=SM+RAIN+tosoil-recharge
                excess=SM-parFC
                excess=torch.clamp(excess,min=0.0)
                SM=SM-excess
                # MODIFY here. Different as class HBVMulT(). Add a ET shape parameter parBETAET
                evapfactor=(SM/(parLP*parFC))**parBETAET
                evapfactor=torch.clamp(evapfactor,min=0.0,max=1.0)
                ETact=EVAP_P*evapfactor
                ETact=torch.min(SM,ETact)
                SM=torch.clamp(SM-ETact,min=PRECS)  # SM can not be zero for gradient tracking

                capillary=torch.min(SLZ,parC*SLZ*(1.0-torch.clamp(SM/parFC,max=1.0)))

                SM=torch.clamp(SM+capillary,min=PRECS)
                SLZ=torch.clamp(SLZ-capillary,min=PRECS)

                # Groundwater boxes
                SUZ=SUZ+recharge+excess
                PERC=torch.min(SUZ,parPERC)
                SUZ=SUZ-PERC
                Q0=parK0*torch.clamp(SUZ-parUZL,min=0.0)
                SUZ=SUZ-Q0
                Q1=parK1*SUZ
                SUZ=SUZ-Q1
                SLZ=SLZ+PERC
                Q2=parK2*SLZ
                SLZ=SLZ-Q2
                Qsimmu[t,:,:]=Q0+Q1+Q2

                # save components for Q
                Qsimmu0[t,:,:]=Q0
                Qsimmu1[t,:,:]=Q1
                Qsimmu2[t,:,:]=Q2
                ETmu[t,:,:]=ETact
                SWEmu[t,:,:]=SNOWPACK
                # Not used, for debug state variables
                # SMlog[t,:, :] = SM.detach().cpu().numpy()
                # SUZlog[t,:,:] = SUZ.detach().cpu().numpy()
                # SLZlog[t,:,:] = SLZ.detach().cpu().numpy()

                SUZ=torch.clamp(SUZ-wateruse[bufftime:][t]*routc2,min=PRECS)
                SLZ=torch.clamp(SLZ-wateruse[bufftime:][t]*routc3,min=PRECS)
                routc1_all[t]=routc1

        Qsimave0=Qsimmu0.mean(-1,keepdim=True)
        Qsimave1=Qsimmu1.mean(-1,keepdim=True)
        Qsimave2=Qsimmu2.mean(-1,keepdim=True)
        ETave=ETmu.mean(-1,keepdim=True)
        SWEave=SWEmu.mean(-1,keepdim=True)

        # get the initial average
        if muwts is None:  # simple average
            Qsimave=Qsimmu.mean(-1)
        else:  # weighted average using learned weights
            Qsimave=(Qsimmu*muwts).sum(-1)

        if routOpt is True:  # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim=Qsimmu.view(Nstep,Ngrid*mu)
            else:
                # average the components, then do routing
                Qsim=Qsimave

            # scale two routing parameters
            tempa=routscaLst[0][0]+rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb=routscaLst[1][0]+rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])

            routa=tempa.repeat(Nstep,1).unsqueeze(-1)
            routb=tempb.repeat(Nstep,1).unsqueeze(-1)

            UH=UH_gamma(routa,routb,lenF=15)  # lenF: folter

            Qsim_loss = torch.clamp(Qsim.unsqueeze(-1)-wateruse[bufftime:]*routc1_all,min=PRECS).squeeze()
            rf=torch.unsqueeze(Qsim_loss,-1).permute([1,2,0])  # dim:gage*var*time
            UH=UH.permute([1,2,0])  # dim: gage*var*time
            # Qsrout=UH_conv(rf,UH).permute([2,0,1])-wateruse[bufftime:]*routc
            # pjli relu function
            Qsrout=UH_conv(rf,UH).permute([2,0,1])

            if comprout is True:  # Qs is [time, [gage*mult], var] now
                Qstemp=Qsrout.view(Nstep,Ngrid,mu)
                if muwts is None:
                    Qs=Qstemp.mean(-1,keepdim=True)
                else:
                    Qs=(Qstemp*muwts).sum(-1,keepdim=True)
            else:
                Qs=Qsrout

        else:  # no routing, output the initial average simulations
            Qs=torch.unsqueeze(Qsimave,-1)  # add a dimension

        if outstate is True:  # output states
            return Qs,SNOWPACK,MELTWATER,SM,SUZ,SLZ
        else:
            # return Qs
            Qall=torch.cat((Qs,Qsimave0,Qsimave1,Qsimave2,ETave,SWEave),dim=-1)
            return Qall[bufftime_old:]

def reservoir_outflow(I, V_tot, theta_FSCR, theta_corr, theta_k, attr_res, dt):
    """
    Calculate reservoir outflow Q based on time-series inflow I, initial storage V, and parameters in p.

    Parameters:
    - I: torch.Tensor of inflows (nt, nb) where nt is the number of timesteps and nb is the batch size.
    - V_init: torch.Tensor of initial reservoir storage volumes (nb,).
    - p: torch.Tensor of parameters (nb, np), where np represents the number of parameters
         (Q_f, Q_n, V_e, V_f, V_c, V_tot, k).

    - V_tot: maximum of reservervoir storage [0,1000] [0,500]
    - th_fscr: the flood storage capacity ratio [0,1]
    - th_corr: the parameter is to correct the 100_simulated flood [1.1,1.6]
    - Q_f and Q_n from the attr_res, Q_f: 30% of magnitude of 100-year flooding, Q_n: mean of flow
    - k: release coefficAAient [0,1]

    Returns:
    - Q: torch.Tensor of calculated outflows (nt, nb)
    - V: torch.Tensor of storage volumes (nt, nb)
    """
    # Unpack parameters from p
    # Q_f, Q_n, V_e, V_f, V_c, V_tot, k = p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 4], p[:, 5], p[:, 6]
    Q_f = attr_res[:,0]*theta_corr*0.3
    Q_n = attr_res[:,1]

    V_f=V_tot*(1.0-theta_FSCR)
    V_e=V_f+0.8*(V_tot-V_f)
    V_c=0.5*V_f
    k = theta_k
    V_init = V_c


    # Initialize output tensors
    nt, nb = I.shape
    print(nt, nb)
    Q = (torch.zeros([nt, nb],dtype=torch.float32)+0.001).to(I)
    V = (torch.zeros([nt, nb],dtype=torch.float32)+0.001).to(I)
    # I_t = (torch.zeros([nb],dtype=torch.float32)+0.001).to(I)
    # V_t = (torch.zeros([nb],dtype=torch.float32)+0.001).to(I)
    V[0, :] = V_init.clone()  # Set initial storage volume

    for t in range(int(nt/1)):
    #for t in range(500):
        # Current inflow and storage
        I_t = I[t]
        V_t = V[t].clone()

        # Condition masks
        above_design_flood = I_t >= Q_f
        below_design_flood = I_t < Q_f
        overtopping_prevention = (V_e <= V_t) & (V_t <= V_tot)
        flood_control = (V_f <= V_t) & (V_t < V_e)
        water_supply = (V_c <= V_t) & (V_t < V_f)
        water_use_regulation = V_t < V_c

        # Initialize outflow tensor for current time step
        Q_t = (torch.zeros(nb,dtype=torch.float32)+0.001).to(I)

        # Case A: Above design flood discharge
        Q_t[above_design_flood&overtopping_prevention]=I_t[above_design_flood&overtopping_prevention]
        Q_t[above_design_flood&flood_control]=(
                Q_f+k*((V_t-V_f)/(V_e-V_f))*(I_t-Q_f)
        )[above_design_flood&flood_control]
        Q_t[above_design_flood&water_supply]=(
                0.5*Q_n+((V_t-V_c)/(V_f-V_c))*(Q_f-Q_n)
        )[above_design_flood&water_supply]
        Q_t[above_design_flood&water_use_regulation]=(
                Q_n*(V_t/V_f)
        )[above_design_flood&water_use_regulation]

        # Case B: Below design flood discharge
        Q_t[below_design_flood&overtopping_prevention]=Q_f[below_design_flood&overtopping_prevention]
        Q_t[below_design_flood&flood_control]=(
                0.5*Q_n+((V_t-V_c)/(V_e-V_c))**2*(Q_f-Q_n)
        )[below_design_flood&flood_control]
        Q_t[below_design_flood&water_supply]=(
                0.5*Q_n+((V_t-V_c)/(V_e-V_c))**2*(Q_f-Q_n)
        )[below_design_flood&water_supply]
        Q_t[below_design_flood&water_use_regulation]=(
                Q_n*(V_t/V_f)
        )[below_design_flood&water_use_regulation]

        # Condition 1: If V_t + (I_t - Q_t) > V_tot, update Q_t = V_t + I_t - V_tot + 0.001
        cond1=V_t+(I_t-Q_t)>V_tot
        Q_t=torch.where(cond1,V_t+I_t-V_tot+0.001,Q_t)

        # Condition 2: If V_t + (I_t - Q_t) < 0, update Q_t = V_t + I_t - 0.001
        cond2=V_t+(I_t-Q_t)<0
        Q_t=torch.where(cond2,V_t+I_t-0.001,Q_t)

        # Condition 3: Q_t<0, update Q_t = 0
        cond3=Q_t<0
        Q_t=torch.where(cond3,torch.zeros([nb],dtype=torch.float32).to(I),Q_t)

        # Save results
        Q[t] = Q_t
        if t < nt - 1:
            # Update storage for the next time step
            # Update storage for the next time step, ensure no in-place modifications

            V[t + 1] = V_t + (I_t - Q_t) * 1  # Example storage update; adjust based on your model
        # print(Q_t.detach().cpu().numpy(),I_t.detach().cpu().numpy(),V_t.detach().cpu().numpy())
        #print(Q_t, V_t)
    return Q, V

class HBVMulTDET_1_1p_wateruse_u5_only_reservoir(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""

    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first
    # not only consider Q bust also soil water, upper and lower groundwater
    # all these coefficient sum up 1
    # remove the consideration of soil water and abstract the water use before the implicit routing
    # add a reservoir module
    # Hanazaki, Risa, Dai Yamazaki, and Kei Yoshimura. “Development of a Reservoir Flood Control Scheme for Global Flood Models.” Journal of Advances in Modeling Earth Systems 14, no. 3 (2022): e2021MS002944. https://doi.org/10.1029/2021MS002944.


    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMulTDET_1_1p_wateruse_u5_only_reservoir,self).__init__()

    def forward(self,x,parameters,wateruse,staind,tdlst,mu,muwts,rtwts,bufftime=0,outstate=False,routOpt=False,
                comprout=False,dydrop=False):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     X: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2]
        #     staind:use which time step from the learned para time series for static parameters
        #     tdlst: the index list of hbv parameters set as dynamic
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     dydrop: the possibility to drop a dynamic para to static to reduce potential overfitting
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET

        PRECS=1e-5  # keep the numerical calculation stable
        bufftime_old = bufftime
        bufftime=0 # pjli 20240926, donot understand why, comment it
        # # Initialization to warm-up states
        # if bufftime>0:
        #     with torch.no_grad():
        #         xinit=x[0:bufftime,:,:]
        #         initmodel=HBVMulTDET_1_1p_wateruse_u5_only_reservoir()
        #         # buffpara=parameters[bufftime-1,:,:,:] # pjli 20240926 original
        #         buffpara=parameters[0:bufftime,:,:]  # pql5336, dynamic parameter warm up
        #         Qsinit,SNOWPACK,MELTWATER,SM,SUZ,SLZ,V_init=initmodel(xinit,buffpara,wateruse[0:bufftime],staind,tdlst,mu,
        #                                                        muwts,rtwts,
        #                                                        bufftime=0,outstate=True,routOpt=False,comprout=False,
        #                                                        dydrop=False)
        # else:

        # Without warm-up bufftime=0, initialize state variables with zeros
        Ngrid=x.shape[1]
        SNOWPACK=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        MELTWATER=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SM=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SUZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SLZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        # V_init=(torch.zeros([x.shape[0],Ngrid],dtype=torch.float32)+0.001).to(x)
        # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(x)

        P=x[bufftime:,:,0]
        Pm=P.unsqueeze(2).repeat(1,1,mu)  # precip
        T=x[bufftime:,:,1]
        Tm=T.unsqueeze(2).repeat(1,1,mu)  # temperature
        ETpot=x[bufftime:,:,2]
        ETpm=ETpot.unsqueeze(2).repeat(1,1,mu)  # potential ET
        parAll=parameters[bufftime:,:,:,:]
        parAllTrans=torch.zeros_like(parAll)

        ## scale the parameters to real values from [0,1]
        hbvscaLst=[[1,6],[50,1000],[0.05,0.9],[0.01,0.5],[0.001,0.2],[0.2,1],
                   [0,10],[0,100],[-2.5,2.5],[0.5,10],[0,0.1],[0,0.2],[0.3,5],[0,1],[0.001,500],[0.001,0.999],
                   [1.1,1.6],[0.001,1]]  # HBV para
        routscaLst=[[0,2.9],[0,6.5]]  # routing para

        for ip in range(len(hbvscaLst)):  # not include routing. Scaling the parameters
            parAllTrans[:,:,ip,:]=hbvscaLst[ip][0]+parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])



        Nstep,Ngrid=P.size()

        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        parstaFull=parAllTrans[staind,:,:,:].unsqueeze(0).repeat([Nstep,1,1,1])  # static para matrix
        parhbvFull=torch.clone(parstaFull)
        # create probability mask for each parameter on the basin dimension
        pmat=torch.ones([1,Ngrid,1])*dydrop
        for ix in tdlst:
            staPar=parstaFull[:,:,ix-1,:]
            dynPar=parAllTrans[:,:,ix-1,:]
            drmask=torch.bernoulli(pmat).detach_().to(parhbvFull)  # to drop dynamic parameters as static in some basins
            comPar=dynPar*(1-drmask)+staPar*drmask
            parhbvFull[:,:,ix-1,:]=comPar

        # Initialize time series of model variables to save results
        Qsimmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        ETmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        SWEmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        # Output the box components of Q
        Qsimmu0=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu1=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu2=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)

        # # Not used. Logging the state variables for debug.
        # # SMlog = np.zeros(P.size())
        # logSM = np.zeros(P.size())
        # logPS = np.zeros(P.size())
        # logswet = np.zeros(P.size())
        # logRE = np.zeros(P.size())

        # routc1_all=(torch.zeros(P.size(),dtype=torch.float32)+0.001).to(parhbvFull).unsqueeze(-1)

        # reservoir module
        V_tot_all = (torch.zeros(P.size(),dtype=torch.float32)+0.001).to(parhbvFull).unsqueeze(-1)
        theta_FSCR_all = (torch.zeros(P.size(),dtype=torch.float32)+0.001).to(parhbvFull).unsqueeze(-1)
        theta_corr_all = (torch.zeros(P.size(),dtype=torch.float32)+0.001).to(parhbvFull).unsqueeze(-1)
        theta_k_all = (torch.zeros(P.size(),dtype=torch.float32)+0.001).to(parhbvFull).unsqueeze(-1)

        for t in range(Nstep):
            with torch.no_grad() if t<bufftime_old else torch.enable_grad():
                paraLst=[]

                for ip in range(len(hbvscaLst)):  # unpack HBV parameters
                    paraLst.append(parhbvFull[t,:,ip,:])

                parBETA,parFC,parK0,parK1,parK2,parLP,parPERC,parUZL,parTT,parCFMAX,parCFR,parCWH,parBETAET,parC,\
                    V_tot,theta_FSCR,theta_corr,theta_k =paraLst

                # # water use parameters
                # tempc_all=tempc1+tempc2+tempc3
                # routc1=(tempc1/(tempc_all)).mean(-1).unsqueeze(-1)
                # routc2=(tempc2/(tempc_all)).mean(-1).unsqueeze(-1)
                # routc3=(tempc3/(tempc_all)).mean(-1).unsqueeze(-1)

                # reservoir module parameters
                V_tot_mean = V_tot.mean(-1).unsqueeze(-1)
                theta_FSCR_mean = theta_FSCR.mean(-1).unsqueeze(-1)
                theta_corr_mean = theta_corr.mean(-1).unsqueeze(-1)
                theta_k_mean = theta_k.mean(-1).unsqueeze(-1)



                # Separate precipitation into liquid and solid components
                PRECIP=Pm[t,:,:]
                TEMP=Tm[t,:,:]
                EVAP_P=ETpm[t,:,:]

                RAIN=torch.mul(PRECIP,(TEMP>=parTT).type(torch.float32))
                SNOW=torch.mul(PRECIP,(TEMP<parTT).type(torch.float32))

                # Snow process
                SNOWPACK=SNOWPACK+SNOW
                melt=parCFMAX*(TEMP-parTT)
                melt=torch.clamp(melt,min=0.0)
                melt=torch.min(melt,SNOWPACK)
                MELTWATER=MELTWATER+melt
                SNOWPACK=SNOWPACK-melt
                refreezing=parCFR*parCFMAX*(parTT-TEMP)
                refreezing=torch.clamp(refreezing,min=0.0)
                refreezing=torch.min(refreezing,MELTWATER)
                SNOWPACK=SNOWPACK+refreezing
                MELTWATER=MELTWATER-refreezing
                tosoil=MELTWATER-(parCWH*SNOWPACK)
                tosoil=torch.clamp(tosoil,min=0.0)
                MELTWATER=MELTWATER-tosoil

                # Soil and evaporation
                soil_wetness=(SM/parFC)**parBETA
                # soil_wetness=self.HSC(zd_feature,SM/parFC)
                # soil_wetness=batch_poly_eval(SM/parFC,zd_feature[:,:,:])
                # HSC_input_ori = torch.cat(((SM/parFC)[:,None],zd_feature),dim=1)
                # batch_size,height,width=HSC_input_ori.shape
                # HSC_input = HSC_input_ori.view(batch_size, height*width)
                # soil_wetness = torch.sigmoid(self.HSC(HSC_input))
                soil_wetness=torch.clamp(soil_wetness,min=0.0,max=1.0)
                recharge=(RAIN+tosoil)*soil_wetness

                # Not used, logging states for checking
                # logSM[t,:] = SM.detach().cpu().numpy()
                # logPS[t,:] = (RAIN + tosoil).detach().cpu().numpy()
                # logswet[t,:] = (SM / parFC).detach().cpu().numpy()
                # logRE[t, :] = recharge.detach().cpu().numpy()

                SM=SM+RAIN+tosoil-recharge
                excess=SM-parFC
                excess=torch.clamp(excess,min=0.0)
                SM=SM-excess
                # MODIFY here. Different as class HBVMulT(). Add a ET shape parameter parBETAET
                evapfactor=(SM/(parLP*parFC))**parBETAET
                evapfactor=torch.clamp(evapfactor,min=0.0,max=1.0)
                ETact=EVAP_P*evapfactor
                ETact=torch.min(SM,ETact)
                SM=torch.clamp(SM-ETact,min=PRECS)  # SM can not be zero for gradient tracking

                capillary=torch.min(SLZ,parC*SLZ*(1.0-torch.clamp(SM/parFC,max=1.0)))

                SM=torch.clamp(SM+capillary,min=PRECS)
                SLZ=torch.clamp(SLZ-capillary,min=PRECS)

                # Groundwater boxes
                SUZ=SUZ+recharge+excess
                PERC=torch.min(SUZ,parPERC)
                SUZ=SUZ-PERC
                Q0=parK0*torch.clamp(SUZ-parUZL,min=0.0)
                SUZ=SUZ-Q0
                Q1=parK1*SUZ
                SUZ=SUZ-Q1
                SLZ=SLZ+PERC
                Q2=parK2*SLZ
                SLZ=SLZ-Q2
                Qsimmu[t,:,:]=Q0+Q1+Q2

                # save components for Q
                Qsimmu0[t,:,:]=Q0
                Qsimmu1[t,:,:]=Q1
                Qsimmu2[t,:,:]=Q2
                ETmu[t,:,:]=ETact
                SWEmu[t,:,:]=SNOWPACK
                # Not used, for debug state variables
                # SMlog[t,:, :] = SM.detach().cpu().numpy()
                # SUZlog[t,:,:] = SUZ.detach().cpu().numpy()
                # SLZlog[t,:,:] = SLZ.detach().cpu().numpy()

                # SUZ=torch.clamp(SUZ-wateruse[bufftime:][t]*routc2,min=PRECS)
                # SLZ=torch.clamp(SLZ-wateruse[bufftime:][t]*routc3,min=PRECS)

                # routc1_all[t]=routc1

                # reservoir module
                V_tot_all[t] = V_tot_mean
                theta_FSCR_all[t] = theta_FSCR_mean
                theta_corr_all[t] = theta_corr_mean
                theta_k_all[t] = theta_k_mean

        Qsimave0=Qsimmu0.mean(-1,keepdim=True)
        Qsimave1=Qsimmu1.mean(-1,keepdim=True)
        Qsimave2=Qsimmu2.mean(-1,keepdim=True)
        ETave=ETmu.mean(-1,keepdim=True)
        SWEave=SWEmu.mean(-1,keepdim=True)

        # get the initial average
        if muwts is None:  # simple average
            Qsimave_ori=Qsimmu.mean(-1)
        else:  # weighted average using learned weights
            Qsimave_ori=(Qsimmu*muwts).sum(-1)

        # reservoir module
        # I, V_init, V_tot, theta_FSCR, theta_corr, theta_k, attr_res, dt
        # Qsimave =Qsimave_ori
        Qsimave, V_out = reservoir_outflow(Qsimave_ori, V_tot_all[-1,:,0], theta_FSCR_all[-1,:,0], theta_corr_all[-1,:,0], theta_k_all[-1,:,0], wateruse[-1], Qsimave_ori.size(0))


        if routOpt is True:  # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim=Qsimmu.view(Nstep,Ngrid*mu)
            else:
                # average the components, then do routing
                Qsim=Qsimave

            # scale two routing parameters
            tempa=routscaLst[0][0]+rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb=routscaLst[1][0]+rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])

            routa=tempa.repeat(Nstep,1).unsqueeze(-1)
            routb=tempb.repeat(Nstep,1).unsqueeze(-1)

            UH=UH_gamma(routa,routb,lenF=15)  # lenF: folter

            # Qsim_loss = torch.clamp(Qsim.unsqueeze(-1)-wateruse[bufftime:]*routc1_all,min=PRECS).squeeze()
            # rf=torch.unsqueeze(Qsim_loss,-1).permute([1,2,0])  # dim:gage*var*time
            rf=torch.unsqueeze(Qsim,-1).permute([1,2,0])  # dim:gage*var*time
            UH=UH.permute([1,2,0])  # dim: gage*var*time
            # Qsrout=UH_conv(rf,UH).permute([2,0,1])-wateruse[bufftime:]*routc
            # pjli relu function
            Qsrout=UH_conv(rf,UH).permute([2,0,1])

            if comprout is True:  # Qs is [time, [gage*mult], var] now
                Qstemp=Qsrout.view(Nstep,Ngrid,mu)
                if muwts is None:
                    Qs=Qstemp.mean(-1,keepdim=True)
                else:
                    Qs=(Qstemp*muwts).sum(-1,keepdim=True)
            else:
                Qs=Qsrout

        else:  # no routing, output the initial average simulations
            Qs=torch.unsqueeze(Qsimave,-1)  # add a dimension

        if outstate is True:  # output states
            return Qs,SNOWPACK,MELTWATER,SM,SUZ,SLZ,V_out
        else:
            # return Qs
            Flow_in = Qsimave_ori[:,:,None]
            Flow_out = Qsimave[:,:,None]
            V_out_final=V_out[:,:,None]

            Qall=torch.cat((Qs,Qsimave0,Qsimave1,Qsimave2,ETave,SWEave,V_out_final, Flow_in, Flow_out),dim=-1)
            return Qall[bufftime_old:]

class HBVMulTDET_1_1p_wateruse_u5_dyn_before_reservoir(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""

    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first
    # not only consider Q bust also soil water, upper and lower groundwater
    # all these coefficient sum up 1
    # remove the consideration of soil water and abstract the water use before the implicit routing
    # add a reservoir module
    # Hanazaki, Risa, Dai Yamazaki, and Kei Yoshimura. “Development of a Reservoir Flood Control Scheme for Global Flood Models.” Journal of Advances in Modeling Earth Systems 14, no. 3 (2022): e2021MS002944. https://doi.org/10.1029/2021MS002944.


    def __init__(self):
        """Initiate a HBV instance"""
        super(HBVMulTDET_1_1p_wateruse_u5_dyn_before_reservoir,self).__init__()

    def forward(self,x,parameters,wateruse,staind,tdlst,mu,muwts,rtwts,bufftime=0,outstate=False,routOpt=False,
                comprout=False,dydrop=False):
        # Modified from the original numpy version from Beck et al., 2020. (http://www.gloh2o.org/hbv/) which
        # runs the HBV-light hydrological model (Seibert, 2005).
        # NaN values have to be removed from the inputs.
        #
        # Input:
        #     X: dim=[time, basin, var] forcing array with var P(mm/d), T(deg C), PET(mm/d)
        #     parameters: array with parameter values having the following structure and scales:
        #         BETA[1,6]; FC[50,1000]; K0[0.05,0.9]; K1[0.01,0.5]; K2[0.001,0.2]; LP[0.2,1];
        #         PERC[0,10]; UZL[0,100]; TT[-2.5,2.5]; CFMAX[0.5,10]; CFR[0,0.1]; CWH[0,0.2]
        #     staind:use which time step from the learned para time series for static parameters
        #     tdlst: the index list of hbv parameters set as dynamic
        #     mu:number of components; muwts: weights of components if True; rtwts: routing parameters;
        #     bufftime:warm up period; outstate: output state var; routOpt:routing option; comprout:component routing opt
        #     dydrop: the possibility to drop a dynamic para to static to reduce potential overfitting
        #
        #
        # Output, all in mm:
        #     outstate True: output most state variables for warm-up
        #      Qs:simulated streamflow; SNOWPACK:snow depth; MELTWATER:snow water holding depth;
        #      SM:soil storage; SUZ:upper zone storage; SLZ:lower zone storage
        #     outstate False: output the simulated flux array Qall contains
        #      Qs:simulated streamflow=Q0+Q1+Q2; Qsimave0:Q0 component; Qsimave1:Q1 component; Qsimave2:Q2 baseflow componnet
        #      ETave: actual ET

        PRECS=1e-5  # keep the numerical calculation stable
        bufftime_old = bufftime
        bufftime=0 # pjli 20240926, donot understand why, comment it, set to dynamic warm-up
        # # Initialization to warm-up states
        # if bufftime>0:
        #     with torch.no_grad():
        #         xinit=x[0:bufftime,:,:]
        #         initmodel=HBVMulTDET_1_1p_wateruse_loss()
        #         # buffpara=parameters[bufftime-1,:,:,:] # pjli 20240926 original
        #         buffpara = parameters[0:bufftime,:,:] # pql5336, dynamic parameter warm up
        #         Qsinit,SNOWPACK,MELTWATER,SM,SUZ,SLZ=initmodel(xinit,buffpara,wateruse[0:bufftime],staind,tdlst,mu,muwts,rtwts,
        #                                                        bufftime=0,outstate=True,routOpt=False,comprout=False,
        #                                                        dydrop=False)
        # else:

        # Without warm-up bufftime=0, initialize state variables with zeros
        Ngrid=x.shape[1]
        SNOWPACK=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        MELTWATER=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SM=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SUZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SLZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        # ETact = (torch.zeros([Ngrid,mu], dtype=torch.float32) + 0.001).to(x)

        P=x[bufftime:,:,0]
        Pm=P.unsqueeze(2).repeat(1,1,mu)  # precip
        T=x[bufftime:,:,1]
        Tm=T.unsqueeze(2).repeat(1,1,mu)  # temperature
        ETpot=x[bufftime:,:,2]
        ETpm=ETpot.unsqueeze(2).repeat(1,1,mu)  # potential ET
        parAll=parameters[bufftime:,:,:,:]
        parAllTrans=torch.zeros_like(parAll)

        ## scale the parameters to real values from [0,1]
        hbvscaLst=[[0.5,1.5],[50,1000],[0.05,0.9],[0.01,0.5],[0.001,0.2],[0.2,1],
                   [0,10],[0,100],[-2.5,2.5],[0.5,10],[0,0.1],[0,0.2],[0.3,5],[0,1],[0.001,1],[0.001,1],
                   [0.001,1]]  # HBV para
        routscaLst=[[0,2.9],[0,6.5]]  # routing para

        for ip in range(len(hbvscaLst)):  # not include routing. Scaling the parameters
            parAllTrans[:,:,ip,:]=hbvscaLst[ip][0]+parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        Nstep,Ngrid=P.size()

        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        parstaFull=parAllTrans[staind,:,:,:].unsqueeze(0).repeat([Nstep,1,1,1])  # static para matrix
        parhbvFull=torch.clone(parstaFull)
        # create probability mask for each parameter on the basin dimension
        pmat=torch.ones([1,Ngrid,1])*dydrop
        for ix in tdlst:
            staPar=parstaFull[:,:,ix-1,:]
            dynPar=parAllTrans[:,:,ix-1,:]
            drmask=torch.bernoulli(pmat).detach_().to(parhbvFull)  # to drop dynamic parameters as static in some basins
            comPar=dynPar*(1-drmask)+staPar*drmask
            parhbvFull[:,:,ix-1,:]=comPar

        # Initialize time series of model variables to save results
        Qsimmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        ETmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        SWEmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        # Output the box components of Q
        Qsimmu0=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu1=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        Qsimmu2=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)

        # # Not used. Logging the state variables for debug.
        # # SMlog = np.zeros(P.size())
        # logSM = np.zeros(P.size())
        # logPS = np.zeros(P.size())
        # logswet = np.zeros(P.size())
        # logRE = np.zeros(P.size())

        routc1_all=(torch.zeros(P.size(),dtype=torch.float32)+0.001).to(parhbvFull).unsqueeze(-1)
        for t in range(Nstep):
            with torch.no_grad() if t<bufftime_old else torch.enable_grad():
                paraLst=[]

                for ip in range(len(hbvscaLst)):  # unpack HBV parameters
                    paraLst.append(parhbvFull[t,:,ip,:])

                parBETA,parFC,parK0,parK1,parK2,parLP,parPERC,parUZL,parTT,parCFMAX,parCFR,parCWH,parBETAET,parC,\
                    tempc1,tempc2,tempc3 =paraLst

                # water use parameters
                tempc_all=tempc1+tempc2+tempc3
                routc1=(tempc1/(tempc_all)).mean(-1).unsqueeze(-1)
                routc2=(tempc2/(tempc_all)).mean(-1).unsqueeze(-1)
                routc3=(tempc3/(tempc_all)).mean(-1).unsqueeze(-1)


                # Separate precipitation into liquid and solid components
                PRECIP=Pm[t,:,:]
                TEMP=Tm[t,:,:]
                EVAP_P=ETpm[t,:,:]

                RAIN=torch.mul(PRECIP,(TEMP>=parTT).type(torch.float32))
                SNOW=torch.mul(PRECIP,(TEMP<parTT).type(torch.float32))

                # Snow process
                SNOWPACK=SNOWPACK+SNOW
                melt=parCFMAX*(TEMP-parTT)
                melt=torch.clamp(melt,min=0.0)
                melt=torch.min(melt,SNOWPACK)
                MELTWATER=MELTWATER+melt
                SNOWPACK=SNOWPACK-melt
                refreezing=parCFR*parCFMAX*(parTT-TEMP)
                refreezing=torch.clamp(refreezing,min=0.0)
                refreezing=torch.min(refreezing,MELTWATER)
                SNOWPACK=SNOWPACK+refreezing
                MELTWATER=MELTWATER-refreezing
                tosoil=MELTWATER-(parCWH*SNOWPACK)
                tosoil=torch.clamp(tosoil,min=0.0)
                MELTWATER=MELTWATER-tosoil

                # Soil and evaporation
                soil_wetness=(SM/parFC)**parBETA
                # soil_wetness=self.HSC(zd_feature,SM/parFC)
                # soil_wetness=batch_poly_eval(SM/parFC,zd_feature[:,:,:])
                # HSC_input_ori = torch.cat(((SM/parFC)[:,None],zd_feature),dim=1)
                # batch_size,height,width=HSC_input_ori.shape
                # HSC_input = HSC_input_ori.view(batch_size, height*width)
                # soil_wetness = torch.sigmoid(self.HSC(HSC_input))
                soil_wetness=torch.clamp(soil_wetness,min=0.0,max=1.0)
                recharge=(RAIN+tosoil)*soil_wetness

                # Not used, logging states for checking
                # logSM[t,:] = SM.detach().cpu().numpy()
                # logPS[t,:] = (RAIN + tosoil).detach().cpu().numpy()
                # logswet[t,:] = (SM / parFC).detach().cpu().numpy()
                # logRE[t, :] = recharge.detach().cpu().numpy()

                SM=SM+RAIN+tosoil-recharge
                excess=SM-parFC
                excess=torch.clamp(excess,min=0.0)
                SM=SM-excess
                # MODIFY here. Different as class HBVMulT(). Add a ET shape parameter parBETAET
                evapfactor=(SM/(parLP*parFC))**parBETAET
                evapfactor=torch.clamp(evapfactor,min=0.0,max=1.0)
                ETact=EVAP_P*evapfactor
                ETact=torch.min(SM,ETact)
                SM=torch.clamp(SM-ETact,min=PRECS)  # SM can not be zero for gradient tracking

                capillary=torch.min(SLZ,parC*SLZ*(1.0-torch.clamp(SM/parFC,max=1.0)))

                SM=torch.clamp(SM+capillary,min=PRECS)
                SLZ=torch.clamp(SLZ-capillary,min=PRECS)

                # Groundwater boxes
                SUZ=SUZ+recharge+excess
                PERC=torch.min(SUZ,parPERC)
                SUZ=SUZ-PERC
                Q0=parK0*torch.clamp(SUZ-parUZL,min=0.0)
                SUZ=SUZ-Q0
                Q1=parK1*SUZ
                SUZ=SUZ-Q1
                SLZ=SLZ+PERC
                Q2=parK2*SLZ
                SLZ=SLZ-Q2
                Qsimmu[t,:,:]=Q0+Q1+Q2

                # save components for Q
                Qsimmu0[t,:,:]=Q0
                Qsimmu1[t,:,:]=Q1
                Qsimmu2[t,:,:]=Q2
                ETmu[t,:,:]=ETact
                SWEmu[t,:,:]=SNOWPACK
                # Not used, for debug state variables
                # SMlog[t,:, :] = SM.detach().cpu().numpy()
                # SUZlog[t,:,:] = SUZ.detach().cpu().numpy()
                # SLZlog[t,:,:] = SLZ.detach().cpu().numpy()

                SUZ=torch.clamp(SUZ-wateruse[bufftime:][t]*routc2,min=PRECS)
                SLZ=torch.clamp(SLZ-wateruse[bufftime:][t]*routc3,min=PRECS)
                routc1_all[t]=routc1

        Qsimave0=Qsimmu0.mean(-1,keepdim=True)
        Qsimave1=Qsimmu1.mean(-1,keepdim=True)
        Qsimave2=Qsimmu2.mean(-1,keepdim=True)
        ETave=ETmu.mean(-1,keepdim=True)
        SWEave=SWEmu.mean(-1,keepdim=True)

        # get the initial average
        if muwts is None:  # simple average
            Qsimave=Qsimmu.mean(-1)
        else:  # weighted average using learned weights
            Qsimave=(Qsimmu*muwts).sum(-1)

        if routOpt is True:  # routing
            if comprout is True:
                # do routing to all the components, reshape the mat to [Time, gage*multi]
                Qsim=Qsimmu.view(Nstep,Ngrid*mu)
            else:
                # average the components, then do routing
                Qsim=Qsimave

            # scale two routing parameters
            tempa=routscaLst[0][0]+rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
            tempb=routscaLst[1][0]+rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])

            routa=tempa.repeat(Nstep,1).unsqueeze(-1)
            routb=tempb.repeat(Nstep,1).unsqueeze(-1)

            UH=UH_gamma(routa,routb,lenF=15)  # lenF: folter

            Qsim_loss = torch.clamp(Qsim.unsqueeze(-1)-wateruse[bufftime:]*routc1_all,min=PRECS).squeeze()
            rf=torch.unsqueeze(Qsim_loss,-1).permute([1,2,0])  # dim:gage*var*time
            UH=UH.permute([1,2,0])  # dim: gage*var*time
            # Qsrout=UH_conv(rf,UH).permute([2,0,1])-wateruse[bufftime:]*routc
            # pjli relu function
            Qsrout=UH_conv(rf,UH).permute([2,0,1])

            if comprout is True:  # Qs is [time, [gage*mult], var] now
                Qstemp=Qsrout.view(Nstep,Ngrid,mu)
                if muwts is None:
                    Qs=Qstemp.mean(-1,keepdim=True)
                else:
                    Qs=(Qstemp*muwts).sum(-1,keepdim=True)
            else:
                Qs=Qsrout

        else:  # no routing, output the initial average simulations
            Qs=torch.unsqueeze(Qsimave,-1)  # add a dimension

        if outstate is True:  # output states
            return Qs,SNOWPACK,MELTWATER,SM,SUZ,SLZ
        else:
            # return Qs
            Qall=torch.cat((Qs,Qsimave0,Qsimave1,Qsimave2,ETave,SWEave),dim=-1)
            return Qall[bufftime_old:]

class MultiInv_HBVTDModel(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z)  # dim: Time, Gage, Para
        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_u1(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_u1,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET_1_1p()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z)  # dim: Time, Gage, Para
        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out

class MultiInv_HBVTDModel_u12(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    # 1.1p no gradient
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_u12,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET_1_1p_no_g()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z)  # dim: Time, Gage, Para
        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out
    
    
class MultiInv_HBVTDModel_u13(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    # 1.1p no gradient
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_u13,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET_1_1p_no_g_05()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z)  # dim: Time, Gage, Para
        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out

class MultiInv_HBVTDModel_u30(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    # 1.1p no gradient
    # pjli u30++ adjust for output soil zone soil moisutre storage
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False,
                 model_name='HBV_Smap',
                 model_name_sub='HBVMulTDET_1_1p_no_g_SMAP'):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_u30,self).__init__()

        package_name="hydroDL.model.HydroModels"
        model_import_string=f"{package_name}.{model_name}"

        try:
            PBMmodel=getattr(importlib.import_module(model_import_string),
                             model_name_sub)
        except ImportError:
            print(f"Failed to import {model_import_string} from {model_name}")


        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=PBMmodel(nmul)
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z)  # dim: Time, Gage, Para
        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_u2(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    # 1.1p + P_bias
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_u2,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET_1_1p_PE_bias()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z)  # dim: Time, Gage, Para
        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out

class MultiInv_HBVTDModel_u3(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    # 1.1p + P_bias + E_bias
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_u3,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET_1_1p_PE_bias2()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z)  # dim: Time, Gage, Para
        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out

class MultiInv_HBVTDModel_u4(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    # 1.1p + P_bias + E_bias
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_u4,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET_1_1p_PE_bias3()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z)  # dim: Time, Gage, Para
        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_u5(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    # 1.1p + P_bias + E_bias + two t
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_u5,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET_1_1p_PE_bias4()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z)  # dim: Time, Gage, Para
        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_u6(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    # 1.1p + lstm --> update  prcp
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_u6,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstmfor=CudnnLstmModel(
            nx=1,ny=1,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET_1_1p()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z)  # dim: Time, Gage, Para
        prcp_update = F.relu(self.lstmfor(x[:,:,0,None]))
        x_update = torch.cat((prcp_update,x[:,:,1:]),dim=2)
        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x_update,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out

class MultiInv_HBVTDModel_u7(torch.nn.Module):
    # only use HAND feature to simulate UZL
    # class for dPL + HBV with multiple components and some dynamic parameters
    # original version, only water use is considered
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,
                 drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_u7,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=3
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm


        self.lstminv=CudnnLstmModel(
            nx=self.ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET_1_1p_wateruse()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        ntstep=z.shape[0]
        ngage=z.shape[1]


        Params0=self.lstminv(z)  # dim: Time, Gage, Para


        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul) # # hbv scaled para, [0,1]
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        routpara0=Params0[-1,:,self.nhbvpm-self.nmul:self.nhbvpm+self.nroutpm-self.nmul]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x[:,:,:3],parameters=hbvpara, wateruse = x[:,:,3:], staind=self.staind, tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out

class MultiInv_HBVTDModel_u7(torch.nn.Module):
    # only use HAND feature to simulate UZL
    # class for dPL + HBV with multiple components and some dynamic parameters
    # original version, only water use is considered
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,
                 drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_u7,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2+1
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm


        self.lstminv=CudnnLstmModel(
            nx=self.ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET_1_1p_wateruse()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        ntstep=z.shape[0]
        ngage=z.shape[1]


        Params0=self.lstminv(z)  # dim: Time, Gage, Para


        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul) # # hbv scaled para, [0,1]
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        routpara0=Params0[-1,:,self.nhbvpm-self.nmul:self.nhbvpm+self.nroutpm-self.nmul]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x[:,:,:3],parameters=hbvpara, wateruse = x[:,:,3:], staind=self.staind, tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out

class MultiInv_HBVTDModel_u7_ana(torch.nn.Module):
    # only use HAND feature to simulate UZL
    # class for dPL + HBV with multiple components and some dynamic parameters
    # original version, only water use is considered
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,
                 drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_u7_ana,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=3
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm


        self.lstminv=CudnnLstmModel(
            nx=self.ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET_1_1p_wateruse_loss()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        ntstep=z.shape[0]
        ngage=z.shape[1]


        Params0=self.lstminv(z)  # dim: Time, Gage, Para


        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul) # # hbv scaled para, [0,1]
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        routpara0=Params0[-1,:,self.nhbvpm-self.nmul:self.nhbvpm+self.nroutpm-self.nmul]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x[:,:,:3],parameters=hbvpara, wateruse = x[:,:,3:], staind=self.staind, tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out

class MultiInv_HBVTDModel_u8_multi(torch.nn.Module):
    # only use HAND feature to simulate UZL
    # class for dPL + HBV with multiple components and some dynamic parameters
    # original version, only water use is considered
    # not only consider Q but also the soil water, upper and lower groundwater
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,
                 drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_u8_multi,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2+1+3
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm


        self.lstminv=CudnnLstmModel(
            nx=self.ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET_1_1p_wateruse_u2()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        ntstep=z.shape[0]
        ngage=z.shape[1]


        Params0=self.lstminv(z)  # dim: Time, Gage, Para


        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul) # # hbv scaled para, [0,1]
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        routpara0=Params0[-1,:,self.nhbvpm-self.nmul:self.nhbvpm+self.nroutpm-self.nmul]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x[:,:,:3],parameters=hbvpara, wateruse = x[:,:,3:], staind=self.staind, tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out




class MultiInv_HBVTDModel_u8_multi_ana(torch.nn.Module):
    # only use HAND feature to simulate UZL
    # class for dPL + HBV with multiple components and some dynamic parameters
    # original version, only water use is considered
    # not only consider Q but also the soil water, upper and lower groundwater
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,
                 drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_u8_multi_ana,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2+1+3
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm


        self.lstminv=CudnnLstmModel(
            nx=self.ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET_1_1p_wateruse_u2_loss()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        ntstep=z.shape[0]
        ngage=z.shape[1]


        Params0=self.lstminv(z)  # dim: Time, Gage, Para


        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul) # # hbv scaled para, [0,1]
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        routpara0=Params0[-1,:,self.nhbvpm-self.nmul:self.nhbvpm+self.nroutpm-self.nmul]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x[:,:,:3],parameters=hbvpara, wateruse = x[:,:,3:], staind=self.staind, tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out

class MultiInv_HBVTDModel_u9_multi_dyn(torch.nn.Module):
    # only use HAND feature to simulate UZL
    # class for dPL + HBV with multiple components and some dynamic parameters
    # original version, only water use is considered
    # not only consider Q but also the soil water, upper and lower groundwater
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,
                 drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_u9_multi_dyn,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm


        self.lstminv=CudnnLstmModel(
            nx=self.ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET_1_1p_wateruse_u3_dyn()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        ntstep=z.shape[0]
        ngage=z.shape[1]


        Params0=self.lstminv(z)  # dim: Time, Gage, Para


        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul) # # hbv scaled para, [0,1]
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        routpara0=Params0[-1,:,self.nhbvpm-self.nmul:self.nhbvpm+self.nroutpm-self.nmul]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x[:,:,:3],parameters=hbvpara, wateruse = x[:,:,3:], staind=self.staind, tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out

class MultiInv_HBVTDModel_u10_multi_dyn_before(torch.nn.Module):
    # only use HAND feature to simulate UZL
    # class for dPL + HBV with multiple components and some dynamic parameters
    # original version, only water use is considered
    # not only consider Q but also the soil water, upper and lower groundwater
    # remove the consideration of soil water and abstract the water use before the implicit routing
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,
                 drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_u10_multi_dyn_before,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm


        self.lstminv=CudnnLstmModel(
            nx=self.ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET_1_1p_wateruse_u4_dyn_before()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        ntstep=z.shape[0]
        ngage=z.shape[1]


        Params0=self.lstminv(z)  # dim: Time, Gage, Para


        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul) # # hbv scaled para, [0,1]
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        routpara0=Params0[-1,:,self.nhbvpm-self.nmul:self.nhbvpm+self.nroutpm-self.nmul]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x[:,:,:3],parameters=hbvpara, wateruse = x[:,:,3:], staind=self.staind, tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out

class MultiInv_HBVTDModel_u11_only_reservoir(torch.nn.Module):
    # only use HAND feature to simulate UZL
    # class for dPL + HBV with multiple components and some dynamic parameters
    # original version, only water use is considered
    # not only consider Q but also the soil water, upper and lower groundwater
    # remove the consideration of soil water and abstract the water use before the implicit routing
    # add a reservoir module
    # Hanazaki, Risa, Dai Yamazaki, and Kei Yoshimura. “Development of a Reservoir Flood Control Scheme for Global Flood Models.” Journal of Advances in Modeling Earth Systems 14, no. 3 (2022): e2021MS002944. https://doi.org/10.1029/2021MS002944.

    def __init__(self,*,ninv,nfea,nmul,hiddeninv,
                 drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_u11_only_reservoir,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm


        self.lstminv=CudnnLstmModel(
            nx=self.ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET_1_1p_wateruse_u5_only_reservoir()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        ntstep=z.shape[0]
        ngage=z.shape[1]

        Params0=self.lstminv(z[:,:,:-2])  # dim: Time, Gage, Para

        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul) # # hbv scaled para, [0,1]
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        routpara0=Params0[-1,:,self.nhbvpm-self.nmul:self.nhbvpm+self.nroutpm-self.nmul]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x[:,:,:],parameters=hbvpara, wateruse = z[:,:,-2:], staind=self.staind, tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out
    
    
class MultiInv_HBVTDModel_u11_only_reservoir_nmul(torch.nn.Module):
    # only use HAND feature to simulate UZL
    # class for dPL + HBV with multiple components and some dynamic parameters
    # original version, only water use is considered
    # not only consider Q but also the soil water, upper and lower groundwater
    # remove the consideration of soil water and abstract the water use before the implicit routing
    # add a reservoir module
    # Hanazaki, Risa, Dai Yamazaki, and Kei Yoshimura. “Development of a Reservoir Flood Control Scheme for Global Flood Models.” Journal of Advances in Modeling Earth Systems 14, no. 3 (2022): e2021MS002944. https://doi.org/10.1029/2021MS002944.

    def __init__(self,*,ninv,nfea,nmul,hiddeninv,
                 drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False, model_name = 'HBV_reservoir'):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_u11_only_reservoir_nmul,self).__init__()
        
        package_name = "hydroDL.model.HydroModels"
        model_import_string = f"{package_name}.{model_name}"

        try:
            PBMmodel = getattr(importlib.import_module(model_import_string), 'HBVMulTDET_1_1p_wateruse_u5_only_reservoir_nmul')
        except ImportError:
            print(f"Failed to import {model_import_string} from {model_name}")
        
        
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm


        self.lstminv=CudnnLstmModel(
            nx=self.ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=PBMmodel()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        ntstep=z.shape[0]
        ngage=z.shape[1]

        Params0=self.lstminv(z[:,:,:-2])  # dim: Time, Gage, Para

        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul) # # hbv scaled para, [0,1]
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        routpara0=Params0[-1,:,self.nhbvpm-self.nmul:self.nhbvpm+self.nroutpm-self.nmul]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x[:,:,:],parameters=hbvpara, Q_rv = z[:,:,-2:], staind=self.staind, tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_u11_only_reservoir_nmul_NN(torch.nn.Module):
    # only use HAND feature to simulate UZL
    # class for dPL + HBV with multiple components and some dynamic parameters
    # original version, only water use is considered
    # not only consider Q but also the soil water, upper and lower groundwater
    # remove the consideration of soil water and abstract the water use before the implicit routing
    # add a reservoir module
    # Hanazaki, Risa, Dai Yamazaki, and Kei Yoshimura. “Development of a Reservoir Flood Control Scheme for Global Flood Models.” Journal of Advances in Modeling Earth Systems 14, no. 3 (2022): e2021MS002944. https://doi.org/10.1029/2021MS002944.

    def __init__(self,*,ninv,nfea,nmul,hiddeninv,
                 drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False,model_name='HBV_reservoir'):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_u11_only_reservoir_nmul_NN,self).__init__()

        package_name="hydroDL.model.HydroModels"
        model_import_string=f"{package_name}.{model_name}"

        try:
            PBMmodel=getattr(importlib.import_module(model_import_string),
                             'HBVMulTDET_1_1p_wateruse_u5_only_reservoir_nmul_NN')
        except ImportError:
            print(f"Failed to import {model_import_string} from {model_name}")

        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm

        self.lstminv=CudnnLstmModel(
            nx=self.ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=PBMmodel(nmul)
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        ntstep=z.shape[0]
        ngage=z.shape[1]

        Params0=self.lstminv(z[:,:,:])  # dim: Time, Gage, Para

        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul)  # # hbv scaled para, [0,1]
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        routpara0=Params0[-1,:,
                  self.nhbvpm-self.nmul:self.nhbvpm+self.nroutpm-self.nmul]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x[:,:,:],parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,
                     muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out

class MultiInv_HBVTDModel_u11_only_reservoir_nmul_NN2(torch.nn.Module):
    # only use HAND feature to simulate UZL
    # class for dPL + HBV with multiple components and some dynamic parameters
    # original version, only water use is considered
    # not only consider Q but also the soil water, upper and lower groundwater
    # remove the consideration of soil water and abstract the water use before the implicit routing
    # add a reservoir module
    # Hanazaki, Risa, Dai Yamazaki, and Kei Yoshimura. “Development of a Reservoir Flood Control Scheme for Global Flood Models.” Journal of Advances in Modeling Earth Systems 14, no. 3 (2022): e2021MS002944. https://doi.org/10.1029/2021MS002944.
    # compared with NN1, the Q is also scale into [0,1], using Q_rv, need to load Q_rv

    def __init__(self,*,ninv,nfea,nmul,hiddeninv,
                 drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False,model_name='HBV_reservoir',
                 model_name_sub='HBVMulTDET_1_1p_wateruse_u5_only_reservoir_nmul_NN2'):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_u11_only_reservoir_nmul_NN2,self).__init__()

        package_name="hydroDL.model.HydroModels"
        model_import_string=f"{package_name}.{model_name}"

        try:
            PBMmodel=getattr(importlib.import_module(model_import_string),
                             model_name_sub)
        except ImportError:
            print(f"Failed to import {model_import_string} from {model_name}")

        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm

        self.lstminv=CudnnLstmModel(
            nx=self.ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=PBMmodel(nmul)
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        ntstep=z.shape[0]
        ngage=z.shape[1]

        Params0=self.lstminv(z[:,:,:self.ninv])  # dim: Time, Gage, Para

        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul)  # # hbv scaled para, [0,1]
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        routpara0=Params0[-1,:,
                  self.nhbvpm-self.nmul:self.nhbvpm+self.nroutpm-self.nmul]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x[:,:,:],Q_rv_attr=z[-1,:,(self.ninv-z.shape[2]):],parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,
                     muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_u11_only_reservoir_rout(torch.nn.Module):
    # only use HAND feature to simulate UZL
    # class for dPL + HBV with multiple components and some dynamic parameters
    # original version, only water use is considered
    # not only consider Q but also the soil water, upper and lower groundwater
    # remove the consideration of soil water and abstract the water use before the implicit routing
    # add a reservoir module
    # Hanazaki, Risa, Dai Yamazaki, and Kei Yoshimura. “Development of a Reservoir Flood Control Scheme for Global Flood Models.” Journal of Advances in Modeling Earth Systems 14, no. 3 (2022): e2021MS002944. https://doi.org/10.1029/2021MS002944.

    def __init__(self,*,ninv,nfea,nmul,hiddeninv,
                 drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False,model_name='HBV_reservoir',
                 model_name_sub='HBVMulTDET_1_1p_wateruse_u5_only_reservoir_rout'):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_u11_only_reservoir_rout,self).__init__()

        package_name="hydroDL.model.HydroModels"
        model_import_string=f"{package_name}.{model_name}"

        try:
            PBMmodel=getattr(importlib.import_module(model_import_string),
                             model_name_sub)
        except ImportError:
            print(f"Failed to import {model_import_string} from {model_name}")

        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        # reservoir module
        if 1 ==1:
            nroutpm = nroutpm + 4
        ntp=nhbvpm+nroutpm+nwtspm

        self.lstminv=CudnnLstmModel(
            nx=self.ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=PBMmodel()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        ntstep=z.shape[0]
        ngage=z.shape[1]

        Params0=self.lstminv(z[:,:,:self.ninv])  # dim: Time, Gage, Para

        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul)  # # hbv scaled para, [0,1]
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        routpara0=Params0[-1,:,
                  self.nhbvpm-self.nmul:self.nhbvpm+self.nroutpm-self.nmul]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x[:,:,:],parameters=hbvpara,Q_rv=z[:,:,(self.ninv-z.shape[-1]):],staind=self.staind,tdlst=self.tdlst,mu=self.nmul,
                     muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out
    
    
class MultiInv_HBVTDModel_u11_only_reservoir_rout_wateruse(torch.nn.Module):
    # only use HAND feature to simulate UZL
    # class for dPL + HBV with multiple components and some dynamic parameters
    # original version, only water use is considered
    # not only consider Q but also the soil water, upper and lower groundwater
    # remove the consideration of soil water and abstract the water use before the implicit routing
    # add a reservoir module
    # Hanazaki, Risa, Dai Yamazaki, and Kei Yoshimura. “Development of a Reservoir Flood Control Scheme for Global Flood Models.” Journal of Advances in Modeling Earth Systems 14, no. 3 (2022): e2021MS002944. https://doi.org/10.1029/2021MS002944.

    def __init__(self,*,ninv,nfea,nmul,hiddeninv,
                 drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False,model_name='HBV_reservoir'):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_u11_only_reservoir_rout_wateruse,self).__init__()

        package_name="hydroDL.model.HydroModels"
        model_import_string=f"{package_name}.{model_name}"

        try:
            PBMmodel=getattr(importlib.import_module(model_import_string),
                             'HBVMulTDET_1_1p_wateruse_u5_only_reservoir_rout_wateruse')
        except ImportError:
            print(f"Failed to import {model_import_string} from {model_name}")

        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        # reservoir module
        if 1 ==1:
            nroutpm = nroutpm + 4 + 1
        ntp=nhbvpm+nroutpm+nwtspm

        self.lstminv=CudnnLstmModel(
            nx=self.ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=PBMmodel()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        ntstep=z.shape[0]
        ngage=z.shape[1]

        Params0=self.lstminv(z[:,:,:-2])  # dim: Time, Gage, Para

        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul)  # # hbv scaled para, [0,1]
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        routpara0=Params0[-1,:,
                  self.nhbvpm-self.nmul:self.nhbvpm+self.nroutpm-self.nmul]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x[:,:,:3],parameters=hbvpara,Q_rv=z[:,:,-2:],wateruse=x[:,:,3:] ,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,
                     muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out

class MultiInv_HBVTDModel_u11_multi_dyn_before_reservoir(torch.nn.Module):
    # only use HAND feature to simulate UZL
    # class for dPL + HBV with multiple components and some dynamic parameters
    # original version, only water use is considered
    # not only consider Q but also the soil water, upper and lower groundwater
    # remove the consideration of soil water and abstract the water use before the implicit routing
    # add a reservoir module
    # Hanazaki, Risa, Dai Yamazaki, and Kei Yoshimura. “Development of a Reservoir Flood Control Scheme for Global Flood Models.” Journal of Advances in Modeling Earth Systems 14, no. 3 (2022): e2021MS002944. https://doi.org/10.1029/2021MS002944.

    def __init__(self,*,ninv,nfea,nmul,hiddeninv,
                 drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_u11_multi_dyn_before_reservoir,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm


        self.lstminv=CudnnLstmModel(
            nx=self.ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET_1_1p_wateruse_u5_dyn_before_reservoir()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        ntstep=z.shape[0]
        ngage=z.shape[1]


        Params0=self.lstminv(z)  # dim: Time, Gage, Para


        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul) # # hbv scaled para, [0,1]
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        routpara0=Params0[-1,:,self.nhbvpm-self.nmul:self.nhbvpm+self.nroutpm-self.nmul]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x[:,:,:3],parameters=hbvpara, wateruse = x[:,:,3:], staind=self.staind, tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_HAND_CNN1d_uu1(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    # 1.1p + HAND
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_HAND_CNN1d_uu1,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET_1_1p_HAND()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,zd,doDropMC=False):
        Params0=self.lstminv(z)  # dim: Time, Gage, Para
        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,zd,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_landuse_para(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse_para,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z)  # dim: Time, Gage, Para
        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out,hbvpara.mean(3),hbvpara0.view(ntstep,ngage,self.nfea,self.nmul).mean(3)


class MultiInv_HBVTDModel_CNN(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_CNN,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.cnn2d=ResNet18()
        self.lstminv=CudnnLstmModel(
            nx=ninv+10,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,zd,doDropMC=False):
        ntstep=z.shape[0]

        topo_hidden=self.cnn2d(zd[:,np.newaxis])
        topo_hidden_time=topo_hidden.repeat([ntstep,1,1])
        z_add=torch.cat((z,topo_hidden_time),dim=2)

        Params0=self.lstminv(z_add)  # dim: Time, Gage, Para

        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_CNNv2(torch.nn.Module):
    # nomalized
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_CNNv2,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.cnn2d=ResNet18v2()
        self.lstminv=CudnnLstmModel(
            nx=ninv+10,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,zd,doDropMC=False):
        ntstep=z.shape[0]

        topo_hidden=self.cnn2d(zd[:,np.newaxis])/1000.0
        topo_hidden_time=topo_hidden.repeat([ntstep,1,1])
        z_add=torch.cat((z,topo_hidden_time),dim=2)

        Params0=self.lstminv(z_add)  # dim: Time, Gage, Para

        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out



class MultiInv_HBVTDModel_CNN_LSTM_v3(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,nx,ny,hiddenSize,dr=0.5):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_CNN_LSTM_v3,self).__init__()
        self.nx=nx
        self.ny=ny
        self.hiddenSize=hiddenSize

        self.cnn2d=ResNet18v2()

        Nf=10+self.nx

        self.lstm=CudnnLstmModel(
            nx=Nf,ny=ny,hiddenSize=hiddenSize,dr=dr)

    def forward(self,x,z,zd,doDropMC=False):
        ntstep=z.shape[0]

        topo_hidden=self.cnn2d(zd[:,np.newaxis])/1000.0
        topo_hidden_time=topo_hidden.repeat([ntstep,1,1])
        z_add=torch.cat((z,topo_hidden_time),dim=2)

        out=self.lstm(z_add)  # dim: Time, Gage, Para
        return out


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)  # Flatten the tensor

class MultiInv_HBVTDModel_HAND_CNN1d_5(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,
                 nobs,nkernel=(10,5),kernelSize=(3,3),stride=(2,1),poolOpt=None,
                 drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_HAND_CNN1d_5,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        nlayer=len(nkernel)
        self.features=nn.Sequential()
        ninchan=2
        Lout=nobs
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        for ii in range(nlayer):
            ConvLayer=CNN1dkernel(
                ninchannel=ninchan,nkernel=nkernel[ii],kernelSize=kernelSize[ii],stride=stride[ii])
            self.features.add_module('CNNLayer%d'%(ii+1),ConvLayer)
            ninchan=nkernel[ii]
            Lout=cnn.calConvSize(lin=Lout,kernel=kernelSize[ii],stride=stride[ii])
            self.features.add_module('Relu%d'%(ii+1),nn.ReLU())
            if poolOpt is not None:
                self.features.add_module('Pooling%d'%(ii+1),nn.MaxPool1d(poolOpt[ii]))
                Lout=cnn.calPoolSize(lin=Lout,kernel=poolOpt[ii])
            self.Ncnnout=int(Lout*nkernel[-1])  # total CNN feature number after convolution

        self.Ncnnout2=5
        self.LinLayer=torch.nn.Linear(self.Ncnnout,self.Ncnnout2)

        Nf=self.Ncnnout2+self.ninv

        self.lstminv=CudnnLstmModel(
            nx=Nf,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,zd,doDropMC=False):
        ntstep=z.shape[0]
        ngage=z.shape[1]

        # zd = ngrid*nVar*nchan add a channel dimension
        z0=self.features(zd)
        # z0 = (ngrid) * nkernel * sizeafterconv
        z0_trans=z0.view(ngage,self.Ncnnout)
        z0_line = self.LinLayer(z0_trans)
        z0_time = z0_line.repeat(ntstep,1,1)



        z_add=torch.cat((z,z0_time),dim=2)
        Params0=self.lstminv(z_add)  # dim: Time, Gage, Para

        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_HAND_CNN1d(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,
                 nobs,nkernel=(10,5),kernelSize=(3,3),stride=(2,1),poolOpt=None,
                 drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_HAND_CNN1d,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        nlayer=len(nkernel)
        self.features=nn.Sequential()
        ninchan=2
        Lout=nobs
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        for ii in range(nlayer):
            ConvLayer=CNN1dkernel(
                ninchannel=ninchan,nkernel=nkernel[ii],kernelSize=kernelSize[ii],stride=stride[ii])
            self.features.add_module('CNNLayer%d'%(ii+1),ConvLayer)
            ninchan=nkernel[ii]
            Lout=cnn.calConvSize(lin=Lout,kernel=kernelSize[ii],stride=stride[ii])
            self.features.add_module('Relu%d'%(ii+1),nn.ReLU())
            if poolOpt is not None:
                self.features.add_module('Pooling%d'%(ii+1),nn.MaxPool1d(poolOpt[ii]))
                Lout=cnn.calPoolSize(lin=Lout,kernel=poolOpt[ii])
        self.Ncnnout=int(Lout*nkernel[-1])  # total CNN feature number after convolution
        Nf=self.Ncnnout+self.ninv

        self.lstminv=CudnnLstmModel(
            nx=Nf,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,zd,doDropMC=False):
        ntstep=z.shape[0]
        ngage=z.shape[1]

        # zd = ngrid*nVar*nchan add a channel dimension
        z0=self.features(zd)
        # z0 = (ngrid) * nkernel * sizeafterconv
        z0_time=z0.view(ngage,self.Ncnnout).repeat(ntstep,1,1)

        z_add=torch.cat((z,z0_time),dim=2)
        Params0=self.lstminv(z_add)  # dim: Time, Gage, Para

        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out

class MultiInv_HBVTDModel_HAND_CNN1d_UZL(torch.nn.Module):
    # only use HAND feature to simulate UZL
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,
                 nobs,nkernel=(10,5),kernelSize=(3,3),stride=(2,1),poolOpt=None,
                 drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_HAND_CNN1d_UZL,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        nlayer=len(nkernel)
        self.features=nn.Sequential()
        ninchan=2
        Lout=nobs
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        for ii in range(nlayer):
            ConvLayer=CNN1dkernel(
                ninchannel=ninchan,nkernel=nkernel[ii],kernelSize=kernelSize[ii],stride=stride[ii])
            self.features.add_module('CNNLayer%d'%(ii+1),ConvLayer)
            ninchan=nkernel[ii]
            Lout=cnn.calConvSize(lin=Lout,kernel=kernelSize[ii],stride=stride[ii])
            self.features.add_module('Relu%d'%(ii+1),nn.ReLU())
            if poolOpt is not None:
                self.features.add_module('Pooling%d'%(ii+1),nn.MaxPool1d(poolOpt[ii]))
                Lout=cnn.calPoolSize(lin=Lout,kernel=poolOpt[ii])
        self.Ncnnout=int(Lout*nkernel[-1])  # total CNN feature number after convolution
        Nf=self.Ncnnout+self.ninv

        self.lstminv=CudnnLstmModel(
            nx=self.ninv,ny=ntp-nmul,hiddenSize=hiddeninv,dr=drinv)
        self.lstminv_uzl=CudnnLstmModel(
            nx=self.Ncnnout,ny=nmul,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,zd,doDropMC=False):
        ntstep=z.shape[0]
        ngage=z.shape[1]

        # zd = ngrid*nVar*nchan add a channel dimension
        z0=self.features(zd)
        # z0 = (ngrid) * nkernel * sizeafterconv
        z0_time=z0.view(ngage,self.Ncnnout).repeat(ntstep,1,1)

        z_add=torch.cat((z,z0_time),dim=2)
        Params0=self.lstminv(z)  # dim: Time, Gage, Para
        Params1=self.lstminv_uzl(z0_time)


        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm-self.nmul]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_old=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea-1,self.nmul)  # hbv scaled para, [0,1]
        hbvpara_uzl=torch.sigmoid(Params1).view(ntstep,ngage,1,self.nmul)
        hbvpara = torch.cat((hbvpara_old[:,:,:7],hbvpara_uzl,hbvpara_old[:,:,7:]),dim=2)
        routpara0=Params0[-1,:,self.nhbvpm-self.nmul:self.nhbvpm+self.nroutpm-self.nmul]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out

class MultiInv_HBVTDModel_HAND_CNN1d_6coeff(torch.nn.Module):
    # only use HAND feature to simulate UZL
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,
                 nobs,nkernel=(10,5),kernelSize=(3,3),stride=(2,1),poolOpt=None,
                 drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_HAND_CNN1d_6coeff,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        nlayer=len(nkernel)
        self.features=nn.Sequential()
        ninchan=2
        Lout=nobs
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        for ii in range(nlayer):
            ConvLayer=CNN1dkernel(
                ninchannel=ninchan,nkernel=nkernel[ii],kernelSize=kernelSize[ii],stride=stride[ii])
            self.features.add_module('CNNLayer%d'%(ii+1),ConvLayer)
            ninchan=nkernel[ii]
            Lout=cnn.calConvSize(lin=Lout,kernel=kernelSize[ii],stride=stride[ii])
            self.features.add_module('Relu%d'%(ii+1),nn.ReLU())
            if poolOpt is not None:
                self.features.add_module('Pooling%d'%(ii+1),nn.MaxPool1d(poolOpt[ii]))
                Lout=cnn.calPoolSize(lin=Lout,kernel=poolOpt[ii])
        self.Ncnnout=int(Lout*nkernel[-1])  # total CNN feature number after convolution
        Nf=self.Ncnnout+self.ninv

        self.lstminv=CudnnLstmModel(
            nx=self.ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)
        self.lstminv_coeff=CudnnLstmModel(
            nx=Nf,ny=6*nmul,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET_1_1p_HAND_coeff(nmul)
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,zd,doDropMC=False):
        ntstep=z.shape[0]
        ngage=z.shape[1]

        # zd = ngrid*nVar*nchan add a channel dimension
        z0=self.features(zd)
        # z0 = (ngrid) * nkernel * sizeafterconv
        z0_time=z0.view(ngage,self.Ncnnout).repeat(ntstep,1,1)

        z_add=torch.cat((z,z0_time),dim=2)
        Params0=self.lstminv(z)  # dim: Time, Gage, Para
        Params1=self.lstminv_coeff(z_add)


        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul) # # hbv scaled para, [0,1]
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_coeff=(torch.sigmoid(Params1).view(ntstep,ngage,6,self.nmul)-0.5)*100
        routpara0=Params0[-1,:,self.nhbvpm-self.nmul:self.nhbvpm+self.nroutpm-self.nmul]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara, para_coeff =hbvpara_coeff, staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out

class MultiInv_HBVTDModel_HAND_CNN1d_6coeff_NN(torch.nn.Module):
    # only use HAND feature to simulate UZL
    # class for dPL + HBV with multiple components and some dynamic parameters
    # 6 coefficient but use NN estimated HSC pjli 20241013
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,
                 nobs,nkernel=(10,10,8,5,2),kernelSize=(10,7,5,3,3),stride=(2,3,3,2,2),poolOpt=None,
                 drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):

        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_HAND_CNN1d_6coeff_NN,self).__init__()

        package_name="hydroDL.model.HydroModels"
        model_name="HBVMulTDET_1_1p_HAND"
        model_import_string=f"{package_name}.{model_name}"
        try:
            PBMmodel = getattr(importlib.import_module(model_import_string), 'HBVMulTDET_1_1p_HAND_coeff_NN')
        except ImportError:
            print(f"Failed to import {model_import_string} from {model_name}")


        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        nlayer=len(nkernel)
        self.features=nn.Sequential()
        ninchan=2
        Lout=nobs
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        for ii in range(nlayer):
            ConvLayer=CNN1dkernel(
                ninchannel=ninchan,nkernel=nkernel[ii],kernelSize=kernelSize[ii],stride=stride[ii])
            self.features.add_module('CNNLayer%d'%(ii+1),ConvLayer)
            ninchan=nkernel[ii]
            Lout=cnn.calConvSize(lin=Lout,kernel=kernelSize[ii],stride=stride[ii])
            self.features.add_module('Relu%d'%(ii+1),nn.ReLU())
            if poolOpt is not None:
                self.features.add_module('Pooling%d'%(ii+1),nn.MaxPool1d(poolOpt[ii]))
                Lout=cnn.calPoolSize(lin=Lout,kernel=poolOpt[ii])
        self.Ncnnout=int(Lout*nkernel[-1])  # total CNN feature number after convolution
        Nf=self.Ncnnout+self.ninv

        self.lstminv=CudnnLstmModel(
            nx=self.ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)
        self.lstminv_coeff=CudnnLstmModel(
            nx=Nf,ny=6*nmul,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=PBMmodel(nmul)
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,zd,doDropMC=False):
        ntstep=z.shape[0]
        ngage=z.shape[1]

        # zd = ngrid*nVar*nchan add a channel dimension
        z0=self.features(zd)
        # z0 = (ngrid) * nkernel * sizeafterconv
        z0_time=z0.view(ngage,self.Ncnnout).repeat(ntstep,1,1)

        z_add=torch.cat((z,z0_time),dim=2)
        Params0=self.lstminv(z)  # dim: Time, Gage, Para
        Params1=self.lstminv_coeff(z_add)


        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul) # # hbv scaled para, [0,1]
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_coeff=(torch.sigmoid(Params1).view(ntstep,ngage,6,self.nmul)-0.5)
        routpara0=Params0[-1,:,self.nhbvpm-self.nmul:self.nhbvpm+self.nroutpm-self.nmul]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara, para_coeff =hbvpara_coeff, staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_HAND_CNN1d_6coeff_NN_oneLSTM(torch.nn.Module):
    # only use HAND feature to simulate UZL
    # class for dPL + HBV with multiple components and some dynamic parameters
    # 6 coefficient but use NN estimated HSC pjli 20241013
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,
                 nobs,nkernel=(10,10,8,5,2),kernelSize=(10,7,5,3,3),stride=(2,3,3,2,2),poolOpt=None,
                 drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):

        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_HAND_CNN1d_6coeff_NN_oneLSTM,self).__init__()

        package_name="hydroDL.model.HydroModels"
        model_name="HBVMulTDET_1_1p_HAND"
        model_import_string=f"{package_name}.{model_name}"
        try:
            PBMmodel=getattr(importlib.import_module(model_import_string),'HBVMulTDET_1_1p_HAND_coeff_NN_oneLSTM')
        except ImportError:
            print(f"Failed to import {model_import_string} from {model_name}")

        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        nlayer=len(nkernel)
        self.features=nn.Sequential()
        ninchan=2
        Lout=nobs
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        for ii in range(nlayer):
            ConvLayer=CNN1dkernel(
                ninchannel=ninchan,nkernel=nkernel[ii],kernelSize=kernelSize[ii],stride=stride[ii])
            self.features.add_module('CNNLayer%d'%(ii+1),ConvLayer)
            ninchan=nkernel[ii]
            Lout=cnn.calConvSize(lin=Lout,kernel=kernelSize[ii],stride=stride[ii])
            self.features.add_module('Relu%d'%(ii+1),nn.ReLU())
            self.features.add_module('Dropout%d'%(ii+1),nn.Dropout(p=drinv))
            if poolOpt is not None:
                self.features.add_module('Pooling%d'%(ii+1),nn.MaxPool1d(poolOpt[ii]))
                Lout=cnn.calPoolSize(lin=Lout,kernel=poolOpt[ii])
        self.Ncnnout=int(Lout*nkernel[-1])  # total CNN feature number after convolution
        # (300,2) --> (2,2)
        Nhand=self.Ncnnout+self.ninv

        # self.lstminv=CudnnLstmModel(
        #     nx=self.ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)
        self.lstminv=CudnnLstmModel(
            nx=Nhand,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=PBMmodel(nmul)
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,zd,doDropMC=False):
        ntstep=z.shape[0]
        ngage=z.shape[1]

        # zd = ngrid*nVar*nchan add a channel dimension
        z0=self.features(zd)
        # z0 = (ngrid) * nkernel * sizeafterconv
        z0_time=z0.view(ngage,self.Ncnnout).repeat(ntstep,1,1)

        z_add=torch.cat((z,z0_time),dim=2)
        # Params0=self.lstminv(z)  # dim: Time, Gage, Para
        Params0=self.lstminv(z_add)

        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul)  # # hbv scaled para, [0,1]
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        # hbvpara_coeff=(torch.sigmoid(Params1).view(ntstep,ngage,6,self.nmul)-0.5)
        routpara0=Params0[-1,:,self.nhbvpm-self.nmul:self.nhbvpm+self.nroutpm-self.nmul]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,
                     muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_HAND_CNN1d_10coeff_NN_morefeature_noquantile(torch.nn.Module):
    # only use HAND feature to simulate UZL
    # class for dPL + HBV with multiple components and some dynamic parameters
    # 6 coefficient but use NN estimated HSC pjli 20241013
    # increase feature into 10, and add a full connect layer before output
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,
                 nobs,nkernel = (10, 8, 5, 2, 5),kernelSize = (10, 7, 5, 3, 3),stride = (2, 3, 2, 1, 1),poolOpt = (2, 2, None, None, None), padding = (0, 0, 0, 1, 1),
                 drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_HAND_CNN1d_10coeff_NN_morefeature_noquantile,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        nlayer=len(nkernel)
        self.features=nn.Sequential()
        ninchan=1
        Lout=nobs
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        # Create CNN layers
        for ii in range(nlayer):
            ConvLayer=CNN1dkernel(
                ninchannel=ninchan,nkernel=nkernel[ii],
                kernelSize=kernelSize[ii],stride=stride[ii],padding=padding[ii]  # Pass padding here
            )
            self.features.add_module('CNNLayer%d'%(ii+1),ConvLayer)
            ninchan=nkernel[ii]

            # Update Lout based on convolution
            Lout=cnn.calConvSize(lin=Lout,kernel=kernelSize[ii],stride=stride[ii],padding=padding[ii])

            # # Add ReLU activation
            # self.features.add_module('Relu%d'%(ii+1),nn.ReLU())
            # already added relu function in the customed CNN1dkernel

            # Apply pooling if needed
            if poolOpt is not None and poolOpt[ii] is not None:
                self.features.add_module('Pooling%d'%(ii+1),nn.MaxPool1d(poolOpt[ii]))
                Lout=cnn.calPoolSize(lin=Lout,kernel=poolOpt[ii])

        # After the CNN layers, Lout is the final output length, and nkernel[-1] is the output channels
        self.Ncnnout=int(Lout*nkernel[-1])  # total CNN feature number after convolution

        # Add a fully connected layer (20) to map Ncnnout features to 20
        self.fc=nn.Linear(self.Ncnnout,20)
        Nf=20+self.ninv

        self.lstminv=CudnnLstmModel(
            nx=self.ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)
        self.lstminv_coeff=CudnnLstmModel(
            nx=Nf,ny=10*nmul,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET_1_1p_HAND_coeff_NN_morefeature(nmul)
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,zd,doDropMC=False):
        ntstep=z.shape[0]
        ngage=z.shape[1]

        # zd = ngrid*nVar*nchan add a channel dimension
        z0=self.features(zd)
        # z0 = (ngrid) * nkernel * sizeafterconv
        z0_time=z0.view(ngage,20).repeat(ntstep,1,1)

        z_add=torch.cat((z,z0_time),dim=2)
        Params0=self.lstminv(z)  # dim: Time, Gage, Para
        Params1=self.lstminv_coeff(z_add)


        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul) # # hbv scaled para, [0,1]
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_coeff=(torch.sigmoid(Params1).view(ntstep,ngage,10,self.nmul)-0.5)
        routpara0=Params0[-1,:,self.nhbvpm-self.nmul:self.nhbvpm+self.nroutpm-self.nmul]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara, para_coeff =hbvpara_coeff, staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_HAND_CNN1d_6coeff_NN_lessfeature_noquantile(torch.nn.Module):
    # only use HAND feature to simulate UZL
    # class for dPL + HBV with multiple components and some dynamic parameters
    # 6 coefficient but use NN estimated HSC pjli 20241013
    # increase feature into 10, and add a full connect layer before output
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,
                 nobs,nkernel = (10, 8, 5, 2, 5),kernelSize = (10, 7, 5, 3, 3),stride = (2, 3, 2, 1, 1),poolOpt = (2, 2, None, None, None), padding = (0, 0, 0, 1, 1),
                 drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_HAND_CNN1d_6coeff_NN_lessfeature_noquantile,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        nlayer=len(nkernel)
        self.features=nn.Sequential()
        ninchan=1
        Lout=nobs
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        # Create CNN layers
        for ii in range(nlayer):
            ConvLayer=CNN1dkernel(
                ninchannel=ninchan,nkernel=nkernel[ii],
                kernelSize=kernelSize[ii],stride=stride[ii],padding=padding[ii]  # Pass padding here
            )
            self.features.add_module('CNNLayer%d'%(ii+1),ConvLayer)
            ninchan=nkernel[ii]

            # Update Lout based on convolution
            Lout=cnn.calConvSize(lin=Lout,kernel=kernelSize[ii],stride=stride[ii],padding=padding[ii])

            # # Add ReLU activation
            # self.features.add_module('Relu%d'%(ii+1),nn.ReLU())
            # already added relu function in the customed CNN1dkernel

            # Apply pooling if needed
            if poolOpt is not None and poolOpt[ii] is not None:
                self.features.add_module('Pooling%d'%(ii+1),nn.MaxPool1d(poolOpt[ii]))
                Lout=cnn.calPoolSize(lin=Lout,kernel=poolOpt[ii])

        # After the CNN layers, Lout is the final output length, and nkernel[-1] is the output channels
        self.Ncnnout=int(Lout*nkernel[-1])  # total CNN feature number after convolution

        # Add a flattening layer (combines last two dimensions)
        self.features.add_module('flatten', Flatten())
        # Add a fully connected layer (20) to map Ncnnout features to 6
        self.n_ext = 6
        self.features.add_module('fullyconnect',nn.Linear(self.Ncnnout,self.n_ext))

        Nf=self.n_ext+self.ninv

        self.lstminv=CudnnLstmModel(
            nx=self.ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)
        self.lstminv_coeff=CudnnLstmModel(
            nx=Nf,ny=6*nmul,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET_1_1p_HAND_coeff_NN(nmul)
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,zd,doDropMC=False):
        ntstep=z.shape[0]
        ngage=z.shape[1]

        # zd = ngrid*nVar*nchan add a channel dimension
        z0=self.features(zd)
        # z0 = (ngrid) * nkernel * sizeafterconv
        z0_time=z0.view(ngage,self.n_ext).repeat(ntstep,1,1)

        z_add=torch.cat((z,z0_time),dim=2)
        Params0=self.lstminv(z)  # dim: Time, Gage, Para
        Params1=self.lstminv_coeff(z_add)


        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul) # # hbv scaled para, [0,1]
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_coeff=(torch.sigmoid(Params1).view(ntstep,ngage,self.n_ext,self.nmul)-0.5)
        routpara0=Params0[-1,:,self.nhbvpm-self.nmul:self.nhbvpm+self.nroutpm-self.nmul]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara, para_coeff =hbvpara_coeff, staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_HAND_CNN1d_6coeff_NN_enlarge(torch.nn.Module):
    # only use HAND feature to simulate UZL
    # class for dPL + HBV with multiple components and some dynamic parameters
    # 6 coefficient but use NN estimated HSC pjli 20241013
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,
                 nobs,nkernel=(10,5),kernelSize=(3,3),stride=(2,1),poolOpt=None,
                 drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_HAND_CNN1d_6coeff_NN_enlarge,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        nlayer=len(nkernel)
        self.features=nn.Sequential()
        ninchan=2
        Lout=nobs
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        for ii in range(nlayer):
            ConvLayer=CNN1dkernel(
                ninchannel=ninchan,nkernel=nkernel[ii],kernelSize=kernelSize[ii],stride=stride[ii])
            self.features.add_module('CNNLayer%d'%(ii+1),ConvLayer)
            ninchan=nkernel[ii]
            Lout=cnn.calConvSize(lin=Lout,kernel=kernelSize[ii],stride=stride[ii])
            self.features.add_module('Relu%d'%(ii+1),nn.ReLU())
            if poolOpt is not None:
                self.features.add_module('Pooling%d'%(ii+1),nn.MaxPool1d(poolOpt[ii]))
                Lout=cnn.calPoolSize(lin=Lout,kernel=poolOpt[ii])
        self.Ncnnout=int(Lout*nkernel[-1])  # total CNN feature number after convolution
        Nf=self.Ncnnout+self.ninv

        self.lstminv=CudnnLstmModel(
            nx=self.ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)
        self.lstminv_coeff=CudnnLstmModel(
            nx=Nf,ny=6*nmul,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET_1_1p_HAND_coeff_NN(nmul)
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,zd,doDropMC=False):
        ntstep=z.shape[0]
        ngage=z.shape[1]

        # zd = ngrid*nVar*nchan add a channel dimension
        z0=self.features(zd)
        # z0 = (ngrid) * nkernel * sizeafterconv
        z0_time=z0.view(ngage,self.Ncnnout).repeat(ntstep,1,1)

        z_add=torch.cat((z,z0_time),dim=2)
        Params0=self.lstminv(z)  # dim: Time, Gage, Para
        Params1=self.lstminv_coeff(z_add)


        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul) # # hbv scaled para, [0,1]
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_coeff=(torch.sigmoid(Params1).view(ntstep,ngage,6,self.nmul)-0.5)*5
        routpara0=Params0[-1,:,self.nhbvpm-self.nmul:self.nhbvpm+self.nroutpm-self.nmul]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara, para_coeff =hbvpara_coeff, staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out

class MultiInv_HBVTDModel_HAND_CNN1d_6coeff_NN_remove(torch.nn.Module):
    # only use HAND feature to simulate UZL
    # class for dPL + HBV with multiple components and some dynamic parameters
    # 6 coefficient but use NN estimated HSC pjli 20241013
    # remove some attributes
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,
                 nobs,nkernel=(10,5),kernelSize=(3,3),stride=(2,1),poolOpt=None,
                 drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_HAND_CNN1d_6coeff_NN_remove,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        nlayer=len(nkernel)
        self.features=nn.Sequential()
        ninchan=2
        Lout=nobs
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        for ii in range(nlayer):
            ConvLayer=CNN1dkernel(
                ninchannel=ninchan,nkernel=nkernel[ii],kernelSize=kernelSize[ii],stride=stride[ii])
            self.features.add_module('CNNLayer%d'%(ii+1),ConvLayer)
            ninchan=nkernel[ii]
            Lout=cnn.calConvSize(lin=Lout,kernel=kernelSize[ii],stride=stride[ii])
            self.features.add_module('Relu%d'%(ii+1),nn.ReLU())
            if poolOpt is not None:
                self.features.add_module('Pooling%d'%(ii+1),nn.MaxPool1d(poolOpt[ii]))
                Lout=cnn.calPoolSize(lin=Lout,kernel=poolOpt[ii])
        self.Ncnnout=int(Lout*nkernel[-1])  # total CNN feature number after convolution
        # Nf=self.Ncnnout+self.ninv
        Nf=self.Ncnnout+10

        self.lstminv=CudnnLstmModel(
            nx=self.ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)
        self.lstminv_coeff=CudnnLstmModel(
            nx=Nf,ny=6*nmul,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET_1_1p_HAND_coeff_NN(nmul)
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,zd,doDropMC=False):
        ntstep=z.shape[0]
        ngage=z.shape[1]

        # zd = ngrid*nVar*nchan add a channel dimension
        z0=self.features(zd)
        # z0 = (ngrid) * nkernel * sizeafterconv
        z0_time=z0.view(ngage,self.Ncnnout).repeat(ntstep,1,1)

        z_add=torch.cat((z[:,:,[0,1,2,3,4,5,6,7,8,11]],z0_time),dim=2)
        Params0=self.lstminv(z)  # dim: Time, Gage, Para
        Params1=self.lstminv_coeff(z_add)


        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul) # # hbv scaled para, [0,1]
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_coeff=(torch.sigmoid(Params1).view(ntstep,ngage,6,self.nmul)-0.5)
        routpara0=Params0[-1,:,self.nhbvpm-self.nmul:self.nhbvpm+self.nroutpm-self.nmul]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara, para_coeff =hbvpara_coeff, staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out

class MultiInv_HBVTDModel_u7_6coeff_NN_noHAND(torch.nn.Module):
    # only use HAND feature to simulate UZL
    # class for dPL + HBV with multiple components and some dynamic parameters
    # 6 coefficient but use NN estimated HSC pjli 20241013
    # do not use HAND data, only test effects of NN-based HSC
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,
                 drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_u7_6coeff_NN_noHAND,self).__init__()
        package_name="hydroDL.model.HydroModels"
        model_name="HBVMulTDET_1_1p_HAND"
        model_import_string=f"{package_name}.{model_name}"
        try:
            PBMmodel = getattr(importlib.import_module(model_import_string), 'HBVMulTDET_1_1p_HAND_coeff_NN')
        except ImportError:
            print(f"Failed to import {model_import_string} from {model_name}")

        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        Nf=self.ninv

        self.lstminv=CudnnLstmModel(
            nx=self.ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)
        self.lstminv_coeff=CudnnLstmModel(
            nx=Nf,ny=6*nmul,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=PBMmodel(nmul)
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        ntstep=z.shape[0]
        ngage=z.shape[1]

        # zd = ngrid*nVar*nchan add a channel dimension
        Params0=self.lstminv(z)  # dim: Time, Gage, Para
        Params1=self.lstminv_coeff(z)


        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul) # # hbv scaled para, [0,1]
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_coeff=(torch.sigmoid(Params1).view(ntstep,ngage,6,self.nmul)-0.5)
        routpara0=Params0[-1,:,self.nhbvpm-self.nmul:self.nhbvpm+self.nroutpm-self.nmul]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara, para_coeff =hbvpara_coeff, staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out

class MultiInv_HBVTDModel_u7_6coeff_NN_noHAND_oneLSTM(torch.nn.Module):
    # only use HAND feature to simulate UZL
    # class for dPL + HBV with multiple components and some dynamic parameters
    # 6 coefficient but use NN estimated HSC pjli 20241013
    # do not use HAND data, only test effects of NN-based HSC
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,
                 drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False,model_name_sub='HBVMulTDET_1_1p_HAND_coeff_NN_oneLSTM'):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_u7_6coeff_NN_noHAND_oneLSTM,self).__init__()
        package_name="hydroDL.model.HydroModels"
        model_name="HBVMulTDET_1_1p_HAND"
        model_import_string=f"{package_name}.{model_name}"
        try:
            PBMmodel = getattr(importlib.import_module(model_import_string), model_name_sub)
        except ImportError:
            print(f"Failed to import {model_import_string} from {model_name}")

        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        Nf=self.ninv

        self.lstminv=CudnnLstmModel(
            nx=self.ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=PBMmodel(nmul)
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        ntstep=z.shape[0]
        ngage=z.shape[1]

        # zd = ngrid*nVar*nchan add a channel dimension
        Params0=self.lstminv(z)  # dim: Time, Gage, Para

        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul) # # hbv scaled para, [0,1]
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        routpara0=Params0[-1,:,self.nhbvpm-self.nmul:self.nhbvpm+self.nroutpm-self.nmul]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara, staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_u7_norm(torch.nn.Module):
    # only use HAND feature to simulate UZL
    # class for dPL + HBV with multiple components and some dynamic parameters
    # 6 coefficient but use NN estimated HSC pjli 20241013
    # do not use HAND data, only test effects of NN-based HSC
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,
                 drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False,model_name_sub='HBVMulTDET_1_1p_HAND_coeff_NN_oneLSTM_VG'):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_u7_norm,self).__init__()
        package_name="hydroDL.model.HydroModels"
        model_name="HBVMulTDET_1_1p_HAND"
        model_import_string=f"{package_name}.{model_name}"
        try:
            PBMmodel = getattr(importlib.import_module(model_import_string), model_name_sub)
        except ImportError:
            print(f"Failed to import {model_import_string} from {model_name}")

        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        Nf=self.ninv

        self.lstminv=CudnnLstmModel(
            nx=self.ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=PBMmodel(nmul)
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        ntstep=z.shape[0]
        ngage=z.shape[1]

        # zd = ngrid*nVar*nchan add a channel dimension
        Params0=self.lstminv(z)  # dim: Time, Gage, Para

        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul) # # hbv scaled para, [0,1]
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        routpara0=Params0[-1,:,self.nhbvpm-self.nmul:self.nhbvpm+self.nroutpm-self.nmul]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara, staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out

class MultiInv_HBVTDModel_u7_6coeff_NN_noHAND_slim(torch.nn.Module):
    # only use HAND feature to simulate UZL
    # class for dPL + HBV with multiple components and some dynamic parameters
    # 6 coefficient but use NN estimated HSC pjli 20241013
    # do not use HAND data, only test effects of NN-based HSC
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,
                 drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_u7_6coeff_NN_noHAND_slim,self).__init__()
        package_name="hydroDL.model.HydroModels"
        model_name="HBVMulTDET_1_1p_HAND"
        model_import_string=f"{package_name}.{model_name}"
        try:
            PBMmodel = getattr(importlib.import_module(model_import_string), 'HBVMulTDET_1_1p_HAND_coeff_NN')
        except ImportError:
            print(f"Failed to import {model_import_string} from {model_name}")

        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        Nf=self.ninv

        self.lstminv=CudnnLstmModel(
            nx=self.ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)
        self.lstminv_coeff=CudnnLstmModel(
            nx=Nf,ny=6*nmul,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=PBMmodel(nmul)
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        ntstep=z.shape[0]
        ngage=z.shape[1]

        # zd = ngrid*nVar*nchan add a channel dimension
        Params0=self.lstminv(z)  # dim: Time, Gage, Para
        Params1=self.lstminv_coeff(z)


        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul) # # hbv scaled para, [0,1]
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_coeff=(torch.sigmoid(Params1).view(ntstep,ngage,6,self.nmul)-0.5)
        routpara0=Params0[-1,:,self.nhbvpm-self.nmul:self.nhbvpm+self.nroutpm-self.nmul]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara, para_coeff =hbvpara_coeff, staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out

class MultiInv_HBVTDModel_u7_6coeff_NN_noHAND_poly(torch.nn.Module):
    # only use HAND feature to simulate UZL
    # class for dPL + HBV with multiple components and some dynamic parameters
    # 6 coefficient but use NN estimated HSC pjli 20241013
    # do not use HAND data, only test effects of NN-poly-based HSC
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,
                 drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_u7_6coeff_NN_noHAND_poly,self).__init__()
        package_name="hydroDL.model.HydroModels"
        model_name="HBVMulTDET_1_1p_HAND"
        model_import_string=f"{package_name}.{model_name}"
        try:
            PBMmodel = getattr(importlib.import_module(model_import_string), 'HBVMulTDET_1_1p_HAND_coeff_poly')
        except ImportError:
            print(f"Failed to import {model_import_string} from {model_name}")

        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        Nf=self.ninv

        self.lstminv=CudnnLstmModel(
            nx=self.ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)
        self.lstminv_coeff=CudnnLstmModel(
            nx=Nf,ny=6*nmul,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=PBMmodel(nmul)
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        ntstep=z.shape[0]
        ngage=z.shape[1]

        # zd = ngrid*nVar*nchan add a channel dimension
        Params0=self.lstminv(z)  # dim: Time, Gage, Para
        Params1=self.lstminv_coeff(z)


        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul) # # hbv scaled para, [0,1]
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_coeff=(torch.sigmoid(Params1).view(ntstep,ngage,6,self.nmul)-0.5)
        routpara0=Params0[-1,:,self.nhbvpm-self.nmul:self.nhbvpm+self.nroutpm-self.nmul]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters_old=hbvpara, para_coeff =hbvpara_coeff, staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out

class MultiInv_HBVTDModel_u7_6coeff_NN_noHAND_lambda(torch.nn.Module):
    # only use HAND feature to simulate UZL
    # class for dPL + HBV with multiple components and some dynamic parameters
    # 6 coefficient but use NN estimated HSC pjli 20241013
    # do not use HAND data, only test effects of NN-based HSC
    # also use NN to estimate the evapfactor
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,
                 drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_u7_6coeff_NN_noHAND_lambda,self).__init__()
        package_name="hydroDL.model.HydroModels"
        model_name="HBVMulTDET_1_1p_HAND"
        model_import_string=f"{package_name}.{model_name}"
        try:
            PBMmodel = getattr(importlib.import_module(model_import_string), 'HBVMulTDET_1_1p_HAND_coeff_NN_lambda')
        except ImportError:
            print(f"Failed to import {model_import_string} from {model_name}")

        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        Nf=self.ninv

        self.lstminv=CudnnLstmModel(
            nx=self.ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)
        self.lstminv_coeff1=CudnnLstmModel(
            nx=Nf,ny=6*nmul,hiddenSize=hiddeninv,dr=drinv)
        self.lstminv_coeff2=CudnnLstmModel(
            nx=Nf,ny=6*nmul,hiddenSize=hiddeninv,dr=drinv)


        if ETMod is True:
            # use the added para for ET eq
            self.HBV=PBMmodel(nmul)
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        ntstep=z.shape[0]
        ngage=z.shape[1]

        # zd = ngrid*nVar*nchan add a channel dimension
        Params0=self.lstminv(z)  # dim: Time, Gage, Para
        Params1=self.lstminv_coeff1(z)
        Params2=self.lstminv_coeff2(z)


        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul) # # hbv scaled para, [0,1]
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_coeff1=(torch.sigmoid(Params1).view(ntstep,ngage,6,self.nmul)-0.5)
        hbvpara_coeff2=(torch.sigmoid(Params2).view(ntstep,ngage,6,self.nmul)-0.5)
        routpara0=Params0[-1,:,self.nhbvpm-self.nmul:self.nhbvpm+self.nroutpm-self.nmul]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara, para_coeff1 =hbvpara_coeff1, para_coeff2 =hbvpara_coeff2, staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_HAND_CNN1d_1var(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,
                 nobs,nkernel=(10,5),kernelSize=(3,3),stride=(2,1),poolOpt=None,
                 drinv=0.5,inittime=365,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_HAND_CNN1d_1var,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        nlayer=len(nkernel)
        self.features=nn.Sequential()
        ninchan=1
        Lout=nobs
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        for ii in range(nlayer):
            ConvLayer=CNN1dkernel(
                ninchannel=ninchan,nkernel=nkernel[ii],kernelSize=kernelSize[ii],stride=stride[ii])
            self.features.add_module('CNNLayer%d'%(ii+1),ConvLayer)
            ninchan=nkernel[ii]
            Lout=cnn.calConvSize(lin=Lout,kernel=kernelSize[ii],stride=stride[ii])
            self.features.add_module('Relu%d'%(ii+1),nn.ReLU())
            if poolOpt is not None:
                self.features.add_module('Pooling%d'%(ii+1),nn.MaxPool1d(poolOpt[ii]))
                Lout=cnn.calPoolSize(lin=Lout,kernel=poolOpt[ii])
        self.Ncnnout=int(Lout*nkernel[-1])  # total CNN feature number after convolution
        Nf=self.Ncnnout+self.ninv

        self.lstminv=CudnnLstmModel(
            nx=Nf,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,zd,doDropMC=False):
        ntstep=z.shape[0]
        ngage=z.shape[1]

        # zd = ngrid*nVar*nchan add a channel dimension
        z0=self.features(zd)
        # z0 = (ngrid) * nkernel * sizeafterconv
        z0_time=z0.view(ngage,self.Ncnnout).repeat(ntstep,1,1)

        z_add=torch.cat((z,z0_time),dim=2)
        Params0=self.lstminv(z_add)  # dim: Time, Gage, Para

        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_landuse1(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse1,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,:])  # dim: Time, Gage, Para
        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        return out


class MultiInv_HBVTDModel_landuse2(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse2,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para
        z_new=z[:,:,0,np.newaxis]
        Params0_delta=self.lstminv_landuse(z_new)
        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw[:self.inittime].size()[0]
            hbvpara_raw[:self.inittime]=hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        for ix in self.tdlst:
            hbvpara_raw[self.inittime:,:,ix-1]=hbvpara_ori[self.inittime:,:,ix-1]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw[:self.inittime,:,ix-1]=hbvpara_ori[self.inittime+self.staind,:,ix-1]
        ################################################################################
        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]
        hbvpara_delta=torch.tanh(hbvpara0_delta).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]

        hbvpara=torch.clamp(hbvpara_raw+hbvpara_delta,min=0.0,max=1.0).view(ntstep,ngage,self.nfea,self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_landuse3_2(torch.nn.Module):
    # only beta delta updated
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_2,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        # self.lstminv_landuse = CudnnLstmModel(
        #     nx=1+self.nfea*self.nmul, ny=ntp, hiddenSize=hiddeninv, dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1+self.nmul,ny=self.nmul,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw[:self.inittime].size()[0]
            hbvpara_raw[:self.inittime]=hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        for ix in self.tdlst:
            hbvpara_raw[self.inittime:,:,ix-1]=hbvpara_ori[self.inittime:,:,ix-1]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw[:self.inittime,:,ix-1]=hbvpara_ori[self.inittime+self.staind,:,ix-1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_norm=hbvpara_raw-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        # hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep, ngage, self.nfea*self.nmul)
        hbvpara_raw_norm_reshape=hbvpara_raw_norm[:,:,:1,:].view(ntstep,ngage,self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,0,np.newaxis]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # hbvpara0_delta = Params0_delta[:, :, 0:self.nhbvpm]
        # hbvpara_delta = torch.tanh(hbvpara0_delta).view(ntstep, ngage, self.nfea, self.nmul) # hbv scaled para, [0,1]

        ############ pql20240430
        hbvpara0_delta=Params0_delta[:,:,0:self.nmul]
        hbvpara_delta=torch.tanh(hbvpara0_delta).view(ntstep,ngage,1,self.nmul)  # hbv scaled para, [0,1]

        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # hbvpara = torch.clamp(hbvpara_raw+hbvpara_delta, min=0.0, max=1.0).view(ntstep, ngage, self.nfea, self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        ##############################
        hbvpara_beta=torch.clamp(hbvpara_raw[:,:,:1,:]+hbvpara_delta,min=0.0,max=1.0).view(ntstep,ngage,1,self.nmul)
        hbvpara=torch.clone(hbvpara_raw)
        hbvpara[:,:,:1,:]=hbvpara_beta

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out


def custom_sigmoid(x):
    return x/(1+torch.abs(x))


class MultiInv_HBVTDModel_landuse3_3(torch.nn.Module):
    # only tanh changed to x/(1+|x|)
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1+self.nfea*self.nmul,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw[:self.inittime].size()[0]
            hbvpara_raw[:self.inittime]=hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        for ix in self.tdlst:
            hbvpara_raw[self.inittime:,:,ix-1]=hbvpara_ori[self.inittime:,:,ix-1]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw[:self.inittime,:,ix-1]=hbvpara_ori[self.inittime+self.staind,:,ix-1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_norm=hbvpara_raw-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep,ngage,self.nfea*self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,0,np.newaxis]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]
        hbvpara_delta=custom_sigmoid(hbvpara0_delta).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]

        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara=torch.clamp(hbvpara_raw+hbvpara_delta,min=0.0,max=1.0).view(ntstep,ngage,self.nfea,self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_landuse3_nonorm(torch.nn.Module):
    # no nomalize for initial parameters
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_nonorm,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1+self.nfea*self.nmul,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw[:self.inittime].size()[0]
            hbvpara_raw[:self.inittime]=hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        for ix in self.tdlst:
            hbvpara_raw[self.inittime:,:,ix-1]=hbvpara_ori[self.inittime:,:,ix-1]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw[:self.inittime,:,ix-1]=hbvpara_ori[self.inittime+self.staind,:,ix-1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_mean = torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        # hbvpara_raw_norm = hbvpara_raw - hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw.view(ntstep,ngage,self.nfea*self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,0,np.newaxis]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]
        hbvpara_delta=torch.tanh(hbvpara0_delta).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]

        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara=torch.clamp(hbvpara_raw+hbvpara_delta,min=0.0,max=1.0).view(ntstep,ngage,self.nfea,self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_landuse3_nonorm_lowslope(torch.nn.Module):
    # no nomalize for initial parameters, activate lowslope
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_nonorm_lowslope,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1+self.nfea*self.nmul,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw[:self.inittime].size()[0]
            hbvpara_raw[:self.inittime]=hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        for ix in self.tdlst:
            hbvpara_raw[self.inittime:,:,ix-1]=hbvpara_ori[self.inittime:,:,ix-1]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw[:self.inittime,:,ix-1]=hbvpara_ori[self.inittime+self.staind,:,ix-1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_mean = torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        # hbvpara_raw_norm = hbvpara_raw - hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw.view(ntstep,ngage,self.nfea*self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,0,np.newaxis]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]
        hbvpara_delta=custom_sigmoid(hbvpara0_delta).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]

        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara=torch.clamp(hbvpara_raw+hbvpara_delta,min=0.0,max=1.0).view(ntstep,ngage,self.nfea,self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_landuse3_nonorm_lowslope_noclamp(torch.nn.Module):
    # no nomalize for initial parameters, activate lowslope, substitute the clamp into sigmoid
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_nonorm_lowslope_noclamp,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1+self.nfea*self.nmul,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw[:self.inittime].size()[0]
            hbvpara_raw[:self.inittime]=hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        for ix in self.tdlst:
            hbvpara_raw[self.inittime:,:,ix-1]=hbvpara_ori[self.inittime:,:,ix-1]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw[:self.inittime,:,ix-1]=hbvpara_ori[self.inittime+self.staind,:,ix-1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_mean = torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        # hbvpara_raw_norm = hbvpara_raw - hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw.view(ntstep,ngage,self.nfea*self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,0,np.newaxis]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]
        hbvpara_delta=custom_sigmoid(hbvpara0_delta).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]

        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################
        hbvpara=torch.sigmoid(hbvpara_raw+hbvpara_delta).view(ntstep,ngage,self.nfea,self.nmul)
        # hbvpara = torch.clamp(hbvpara_raw+hbvpara_delta, min=0.0, max=1.0).view(ntstep, ngage, self.nfea, self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_landuse3(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1+self.nfea*self.nmul,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw[:self.inittime].size()[0]
            hbvpara_raw[:self.inittime]=hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        for ix in self.tdlst:
            hbvpara_raw[self.inittime:,:,ix-1]=hbvpara_ori[self.inittime:,:,ix-1]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw[:self.inittime,:,ix-1]=hbvpara_ori[self.inittime+self.staind,:,ix-1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_norm=hbvpara_raw-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep,ngage,self.nfea*self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,0,np.newaxis]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]
        hbvpara_delta=torch.tanh(hbvpara0_delta).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]

        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara=torch.clamp(hbvpara_raw+hbvpara_delta,min=0.0,max=1.0).view(ntstep,ngage,self.nfea,self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_landuse3_paradelta(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_paradelta,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1+self.nfea*self.nmul,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw[:self.inittime].size()[0]
            hbvpara_raw[:self.inittime]=hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        for ix in self.tdlst:
            hbvpara_raw[self.inittime:,:,ix-1]=hbvpara_ori[self.inittime:,:,ix-1]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw[:self.inittime,:,ix-1]=hbvpara_ori[self.inittime+self.staind,:,ix-1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_norm=hbvpara_raw-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep,ngage,self.nfea*self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,0,np.newaxis]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]
        hbvpara_delta=torch.tanh(hbvpara0_delta).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]

        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara=torch.clamp(hbvpara_raw+hbvpara_delta,min=0.0,max=1.0).view(ntstep,ngage,self.nfea,self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out,hbvpara_raw.mean(3),(hbvpara-hbvpara_raw).mean(3)


class MultiInv_HBVTDModel_landuse3_logadd(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_logadd,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1-2,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1+2+self.nfea*self.nmul,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1+2:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw[:self.inittime].size()[0]
            hbvpara_raw[:self.inittime]=hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        for ix in self.tdlst:
            hbvpara_raw[self.inittime:,:,ix-1]=hbvpara_ori[self.inittime:,:,ix-1]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw[:self.inittime,:,ix-1]=hbvpara_ori[self.inittime+self.staind,:,ix-1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_norm=hbvpara_raw-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep,ngage,self.nfea*self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,:3]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]
        hbvpara_delta=torch.tanh(hbvpara0_delta).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]

        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara=torch.clamp(hbvpara_raw+hbvpara_delta,min=0.0,max=1.0).view(ntstep,ngage,self.nfea,self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_landuse3_logadd_beta(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_logadd_beta,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1-2,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1+2+self.nfea*self.nmul,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1+2:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw[:self.inittime].size()[0]
            hbvpara_raw[:self.inittime]=hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        for ix in self.tdlst:
            hbvpara_raw[self.inittime:,:,ix-1]=hbvpara_ori[self.inittime:,:,ix-1]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw[:self.inittime,:,ix-1]=hbvpara_ori[self.inittime+self.staind,:,ix-1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_norm=hbvpara_raw-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep,ngage,self.nfea*self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,:3]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]
        hbvpara_delta=torch.tanh(hbvpara0_delta).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]

        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################
        hbvpara=hbvpara_raw.view(ntstep,ngage,self.nfea,self.nmul)
        hbvpara[:,:,0,:]=torch.clamp(hbvpara_raw+hbvpara_delta,min=0.0,max=1.0).view(ntstep,ngage,self.nfea,self.nmul)[
                         :,:,0,:]
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_landuse_prepare(torch.nn.Module):
    # change lstm to preprocess the temporal lu data, then be fed into general framework
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse_prepare,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1,ny=1,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        z_trans=self.lstminv_landuse(z[:,:,0,np.newaxis])
        z[:,:,0]=z_trans.squeeze()
        Params0=self.lstminv(z[:,:,:])  # dim: Time, Gage, Para
        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3)  # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_landuse3_seperate(torch.nn.Module):
    # A,x,Alu -->beta, A,x --> others
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_seperate,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp-nmul,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=ninv,ny=nmul,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:(self.nhbvpm-self.nmul)]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea-1,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])

        Params0_beta=self.lstminv_landuse(z)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara_beta=torch.sigmoid(Params0_beta).view(ntstep,ngage,1,self.nmul)  # hbv scaled para, [0,1]

        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara=torch.cat((hbvpara_raw,hbvpara_beta),dim=2)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,(self.nhbvpm-self.nmul):(
                                                               self.nhbvpm-self.nmul)+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_landuse3_twoloss_addrouting(torch.nn.Module):
    # loss_a based on initial para + loss_b on updated para, final loss = loss_a + loss_b*2
    # train, crit, script need modified
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_twoloss_addrouting,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1+ntp,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop
        self.ntp=ntp

    def forward(self,x,z,doDropMC=False):
        Params0=torch.sigmoid(self.lstminv(z[:,:,1:]))  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara_raw_all=torch.clone(Params0)
        Nstep=hbvpara_raw_all[self.inittime:].size()[0]
        hbvpara_raw_all[self.inittime:]=Params0[self.staind].unsqueeze(0).repeat([Nstep,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw_all[:self.inittime].size()[0]
            hbvpara_raw_all[:self.inittime]=Params0[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1])
        for ix in self.tdlst:
            hbvpara_raw_all[self.inittime:,:,(ix-1)*self.nmul:ix*self.nmul]=Params0[self.inittime:,:,
                                                                            (ix-1)*self.nmul:ix*self.nmul]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw_all[:self.inittime,:,(ix-1)*self.nmul:ix*self.nmul]=Params0[self.inittime+self.staind,:,
                                                                                (ix-1)*self.nmul:ix*self.nmul]

        # hbvpara0 = Params0[:, :, 0:self.nhbvpm]
        # # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        # hbvpara_ori = torch.sigmoid(hbvpara0).view(ntstep, ngage, self.nfea, self.nmul) # hbv scaled para, [0,1]
        # ##############################################################################
        # # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        # hbvpara_raw = torch.clone(hbvpara_ori)
        # Nstep = hbvpara_raw[self.inittime:].size()[0]
        # hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep, 1, 1, 1])
        # if self.inittime > 0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
        #     Nstep = hbvpara_raw[:self.inittime].size()[0]
        #     hbvpara_raw[:self.inittime] = hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep, 1, 1, 1])
        # for ix in self.tdlst:
        #     hbvpara_raw[self.inittime:,:,ix-1] = hbvpara_ori[self.inittime:,:,ix-1]
        #     if self.inittime > 0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
        #         hbvpara_raw[:self.inittime, :, ix - 1] = hbvpara_ori[self.inittime+self.staind, :, ix - 1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_norm=hbvpara_raw_all-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep,ngage,self.ntp)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,0,np.newaxis]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        hbvpara_delta=torch.tanh(Params0_delta)

        hbvpara_all=torch.clamp(hbvpara_raw_all+hbvpara_delta,min=0.0,max=1.0)

        hbvpara_raw=hbvpara_raw_all[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul)
        hbvpara=hbvpara_all[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul)

        # # ################################################################################
        # # # Delete specific variables
        # # del hbvpara_raw_norm_reshape,z_new
        # # # Manually release GPU memory
        # # torch.cuda.empty_cache()
        # # ################################################################################

        # hbvpara0_delta = Params0_delta[:, :, 0:self.nhbvpm]
        # hbvpara_delta = torch.tanh(hbvpara0_delta).view(ntstep, ngage, self.nfea, self.nmul) # hbv scaled para, [0,1]

        # # ################################################################################
        # # # Delete specific variables
        # # del hbvpara0_delta, Params0_delta
        # # # Manually release GPU memory
        # # torch.cuda.empty_cache()
        # # ################################################################################

        # hbvpara = torch.clamp(hbvpara_raw+hbvpara_delta, min=0.0, max=1.0).view(ntstep, ngage, self.nfea, self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0_raw=hbvpara_raw_all[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]
        routpara0=hbvpara_all[-1,:,
                  self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            # routpara_raw = torch.sigmoid(routpara0_raw) # [0,1]
            # routpara = torch.sigmoid(routpara0) # [0,1]
            routpara_raw=(routpara0_raw)  # [0,1]
            routpara=(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            # routpara_raw = torch.sigmoid(routpara0_raw).view(ngage * self.nmul, 2) # [0,1]
            # routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2) # [0,1]
            routpara_raw=(routpara0_raw).view(ngage*self.nmul,2)  # [0,1]
            routpara=(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts_raw=None
            wts=None
        else:  # weighted average using learned weights
            wtspara_raw=hbvpara_raw_all[-1,:,-self.nwtspm:]
            wtspara=hbvpara_all[-1,:,-self.nwtspm:]
            wts_raw=F.softmax(wtspara_raw,dim=-1)  # softmax to make sure sum to 1
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out1=self.HBV(x,parameters=hbvpara_raw,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts_raw,
                      rtwts=routpara_raw,
                      bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        out2=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                      bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out1,out2


class MultiInv_HBVTDModel_landuse3_twoloss_addrouting_allDT(torch.nn.Module):
    # loss_a based on initial para + loss_b on updated para, final loss = loss_a + loss_b*2
    # train, crit, script need modified
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_twoloss_addrouting_allDT,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1+ntp,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
            self.HBV_all=HBVMulTDET_all()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop
        self.ntp=ntp

    def forward(self,x,z,doDropMC=False):
        Params0=torch.sigmoid(self.lstminv(z[:,:,1:]))  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara_raw_all=torch.clone(Params0)
        Nstep=hbvpara_raw_all[self.inittime:].size()[0]
        hbvpara_raw_all[self.inittime:]=Params0[self.staind].unsqueeze(0).repeat([Nstep,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw_all[:self.inittime].size()[0]
            hbvpara_raw_all[:self.inittime]=Params0[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1])
        for ix in self.tdlst:
            hbvpara_raw_all[self.inittime:,:,(ix-1)*self.nmul:ix*self.nmul]=Params0[self.inittime:,:,
                                                                            (ix-1)*self.nmul:ix*self.nmul]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw_all[:self.inittime,:,(ix-1)*self.nmul:ix*self.nmul]=Params0[self.inittime+self.staind,:,
                                                                                (ix-1)*self.nmul:ix*self.nmul]

        # hbvpara0 = Params0[:, :, 0:self.nhbvpm]
        # # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        # hbvpara_ori = torch.sigmoid(hbvpara0).view(ntstep, ngage, self.nfea, self.nmul) # hbv scaled para, [0,1]
        # ##############################################################################
        # # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        # hbvpara_raw = torch.clone(hbvpara_ori)
        # Nstep = hbvpara_raw[self.inittime:].size()[0]
        # hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep, 1, 1, 1])
        # if self.inittime > 0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
        #     Nstep = hbvpara_raw[:self.inittime].size()[0]
        #     hbvpara_raw[:self.inittime] = hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep, 1, 1, 1])
        # for ix in self.tdlst:
        #     hbvpara_raw[self.inittime:,:,ix-1] = hbvpara_ori[self.inittime:,:,ix-1]
        #     if self.inittime > 0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
        #         hbvpara_raw[:self.inittime, :, ix - 1] = hbvpara_ori[self.inittime+self.staind, :, ix - 1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_norm=hbvpara_raw_all-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep,ngage,self.ntp)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,0,np.newaxis]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        hbvpara_delta=torch.tanh(Params0_delta)

        hbvpara_all=torch.clamp(hbvpara_raw_all+hbvpara_delta,min=0.0,max=1.0)

        hbvpara_raw=hbvpara_raw_all[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul)
        hbvpara=hbvpara_all[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul)

        # # ################################################################################
        # # # Delete specific variables
        # # del hbvpara_raw_norm_reshape,z_new
        # # # Manually release GPU memory
        # # torch.cuda.empty_cache()
        # # ################################################################################

        # hbvpara0_delta = Params0_delta[:, :, 0:self.nhbvpm]
        # hbvpara_delta = torch.tanh(hbvpara0_delta).view(ntstep, ngage, self.nfea, self.nmul) # hbv scaled para, [0,1]

        # # ################################################################################
        # # # Delete specific variables
        # # del hbvpara0_delta, Params0_delta
        # # # Manually release GPU memory
        # # torch.cuda.empty_cache()
        # # ################################################################################

        # hbvpara = torch.clamp(hbvpara_raw+hbvpara_delta, min=0.0, max=1.0).view(ntstep, ngage, self.nfea, self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0_raw=hbvpara_raw_all[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]
        routpara0=hbvpara_all[:,:,
                  self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara_raw=routpara0_raw  # [0,1]
            # routpara = torch.sigmoid(routpara0) # [0,1]
            routpara=routpara0
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            # routpara_raw = torch.sigmoid(routpara0_raw).view(ngage * self.nmul, 2) # [0,1]
            # routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2) # [0,1]
            routpara_raw=(routpara0_raw).view(ngage*self.nmul,2)  # [0,1]
            routpara=(routpara0).view(ntstep,ngage*self.nmul,2)  # [0,1]

        if self.nwtspm==0:  # simple average for multiple components
            wts_raw=None
            wts=None
        else:  # weighted average using learned weights
            wtspara_raw=hbvpara_raw_all[-1,:,-self.nwtspm:]
            wtspara=hbvpara_all[-1,:,-self.nwtspm:]
            wts_raw=F.softmax(wtspara_raw,dim=-1)  # softmax to make sure sum to 1
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out1=self.HBV(x,parameters=hbvpara_raw,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts_raw,
                      rtwts=routpara_raw,
                      bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        out2=self.HBV_all(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,
                          rtwts=routpara,
                          bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out1,out2


class MultiInv_HBVTDModel_landuse3_twoloss_allDT(torch.nn.Module):
    # loss_a based on initial para + loss_b on updated para, final loss = loss_a + loss_b*2, only changed are DT, routing remained
    # train, crit, script need modified
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_twoloss_allDT,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1+ntp,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
            self.HBV_all_noRT=HBVMulTDET_all_noRT()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop
        self.ntp=ntp

    def forward(self,x,z,doDropMC=False):
        Params0=torch.sigmoid(self.lstminv(z[:,:,1:]))  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara_raw_all=torch.clone(Params0)
        Nstep=hbvpara_raw_all[self.inittime:].size()[0]
        hbvpara_raw_all[self.inittime:]=Params0[self.staind].unsqueeze(0).repeat([Nstep,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw_all[:self.inittime].size()[0]
            hbvpara_raw_all[:self.inittime]=Params0[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1])
        for ix in self.tdlst:
            hbvpara_raw_all[self.inittime:,:,(ix-1)*self.nmul:ix*self.nmul]=Params0[self.inittime:,:,
                                                                            (ix-1)*self.nmul:ix*self.nmul]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw_all[:self.inittime,:,(ix-1)*self.nmul:ix*self.nmul]=Params0[self.inittime+self.staind,:,
                                                                                (ix-1)*self.nmul:ix*self.nmul]

        # hbvpara0 = Params0[:, :, 0:self.nhbvpm]
        # # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        # hbvpara_ori = torch.sigmoid(hbvpara0).view(ntstep, ngage, self.nfea, self.nmul) # hbv scaled para, [0,1]
        # ##############################################################################
        # # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        # hbvpara_raw = torch.clone(hbvpara_ori)
        # Nstep = hbvpara_raw[self.inittime:].size()[0]
        # hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep, 1, 1, 1])
        # if self.inittime > 0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
        #     Nstep = hbvpara_raw[:self.inittime].size()[0]
        #     hbvpara_raw[:self.inittime] = hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep, 1, 1, 1])
        # for ix in self.tdlst:
        #     hbvpara_raw[self.inittime:,:,ix-1] = hbvpara_ori[self.inittime:,:,ix-1]
        #     if self.inittime > 0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
        #         hbvpara_raw[:self.inittime, :, ix - 1] = hbvpara_ori[self.inittime+self.staind, :, ix - 1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_norm=hbvpara_raw_all-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep,ngage,self.ntp)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,0,np.newaxis]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        hbvpara_delta=torch.tanh(Params0_delta)

        hbvpara_all=torch.clamp(hbvpara_raw_all+hbvpara_delta,min=0.0,max=1.0)

        hbvpara_raw=hbvpara_raw_all[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul)
        hbvpara=hbvpara_all[:,:,0:self.nhbvpm].view(ntstep,ngage,self.nfea,self.nmul)

        # # ################################################################################
        # # # Delete specific variables
        # # del hbvpara_raw_norm_reshape,z_new
        # # # Manually release GPU memory
        # # torch.cuda.empty_cache()
        # # ################################################################################

        # hbvpara0_delta = Params0_delta[:, :, 0:self.nhbvpm]
        # hbvpara_delta = torch.tanh(hbvpara0_delta).view(ntstep, ngage, self.nfea, self.nmul) # hbv scaled para, [0,1]

        # # ################################################################################
        # # # Delete specific variables
        # # del hbvpara0_delta, Params0_delta
        # # # Manually release GPU memory
        # # torch.cuda.empty_cache()
        # # ################################################################################

        # hbvpara = torch.clamp(hbvpara_raw+hbvpara_delta, min=0.0, max=1.0).view(ntstep, ngage, self.nfea, self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0_raw=hbvpara_raw_all[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]
        # routpara0 = hbvpara_all[:, :, self.nhbvpm:self.nhbvpm+self.nroutpm] # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara_raw=routpara0_raw  # [0,1]
            # routpara = torch.sigmoid(routpara0) # [0,1]
            # routpara = routpara0
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            # routpara_raw = torch.sigmoid(routpara0_raw).view(ngage * self.nmul, 2) # [0,1]
            # routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2) # [0,1]
            routpara_raw=(routpara0_raw).view(ngage*self.nmul,2)  # [0,1]
            # routpara = (routpara0).view(ntstep, ngage * self.nmul, 2) # [0,1]

        if self.nwtspm==0:  # simple average for multiple components
            wts_raw=None
            # wts = None
        else:  # weighted average using learned weights
            wtspara_raw=hbvpara_raw_all[-1,:,-self.nwtspm:]
            # wtspara = hbvpara_all[-1, :, -self.nwtspm:]
            wts_raw=F.softmax(wtspara_raw,dim=-1)  # softmax to make sure sum to 1
            # wts = F.softmax(wtspara, dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out1=self.HBV(x,parameters=hbvpara_raw,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts_raw,
                      rtwts=routpara_raw,
                      bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        out2=self.HBV_all_noRT(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts_raw,
                               rtwts=routpara_raw,
                               bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out1,out2


class MultiInv_HBVTDModel_landuse3_twoloss(torch.nn.Module):
    # loss_a based on initial para + loss_b on updated para, final loss = loss_a + loss_b*2
    # train, crit, script need modified
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_twoloss,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1+self.nfea*self.nmul,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw[:self.inittime].size()[0]
            hbvpara_raw[:self.inittime]=hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        for ix in self.tdlst:
            hbvpara_raw[self.inittime:,:,ix-1]=hbvpara_ori[self.inittime:,:,ix-1]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw[:self.inittime,:,ix-1]=hbvpara_ori[self.inittime+self.staind,:,ix-1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_norm=hbvpara_raw-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep,ngage,self.nfea*self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,0,np.newaxis]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]
        hbvpara_delta=torch.tanh(hbvpara0_delta).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]

        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara=torch.clamp(hbvpara_raw+hbvpara_delta,min=0.0,max=1.0).view(ntstep,ngage,self.nfea,self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out1=self.HBV(x,parameters=hbvpara_raw,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,
                      rtwts=routpara,
                      bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        out2=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                      bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out1,out2


class MultiInv_HBVTDModel_landuse3_twoloss_clamp(torch.nn.Module):
    # loss_a based on initial para + loss_b on updated para, final loss = loss_a + loss_b*2
    # train, crit, script need modified
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_twoloss_clamp,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1+self.nfea*self.nmul,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw[:self.inittime].size()[0]
            hbvpara_raw[:self.inittime]=hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        for ix in self.tdlst:
            hbvpara_raw[self.inittime:,:,ix-1]=hbvpara_ori[self.inittime:,:,ix-1]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw[:self.inittime,:,ix-1]=hbvpara_ori[self.inittime+self.staind,:,ix-1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_norm=hbvpara_raw-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep,ngage,self.nfea*self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,0,np.newaxis]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]
        hbvpara_delta=torch.clamp(hbvpara0_delta,min=-1.0,max=1.0).view(ntstep,ngage,self.nfea,
                                                                        self.nmul)  # hbv scaled para, [0,1]

        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara=torch.clamp(hbvpara_raw+hbvpara_delta,min=0.0,max=1.0).view(ntstep,ngage,self.nfea,self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out1=self.HBV(x,parameters=hbvpara_raw,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,
                      rtwts=routpara,
                      bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        out2=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                      bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out1,out2


class MultiInv_HBVTDModel_landuse3_twoloss_changeweights(torch.nn.Module):
    # loss_a based on initial para + loss_b on updated para, final loss = loss_a + loss_b*weights
    # need inputing a parameter in lossFun
    # train, crit, script need modified
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_twoloss_changeweights,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1+self.nfea*self.nmul,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw[:self.inittime].size()[0]
            hbvpara_raw[:self.inittime]=hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        for ix in self.tdlst:
            hbvpara_raw[self.inittime:,:,ix-1]=hbvpara_ori[self.inittime:,:,ix-1]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw[:self.inittime,:,ix-1]=hbvpara_ori[self.inittime+self.staind,:,ix-1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_norm=hbvpara_raw-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep,ngage,self.nfea*self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,0,np.newaxis]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]
        hbvpara_delta=torch.tanh(hbvpara0_delta).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]

        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara=torch.clamp(hbvpara_raw+hbvpara_delta,min=0.0,max=1.0).view(ntstep,ngage,self.nfea,self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out1=self.HBV(x,parameters=hbvpara_raw,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,
                      rtwts=routpara,
                      bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        out2=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                      bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out1,out2


class MultiInv_HBVTDModel_landuse3_twoloss_changeweights_clamp_narrow(torch.nn.Module):
    # loss_a based on initial para + loss_b on updated para, final loss = loss_a + loss_b*weights
    # need inputing a parameter in lossFun
    # train, crit, script need modified
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_twoloss_changeweights_clamp_narrow,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1+self.nfea*self.nmul,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw[:self.inittime].size()[0]
            hbvpara_raw[:self.inittime]=hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        for ix in self.tdlst:
            hbvpara_raw[self.inittime:,:,ix-1]=hbvpara_ori[self.inittime:,:,ix-1]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw[:self.inittime,:,ix-1]=hbvpara_ori[self.inittime+self.staind,:,ix-1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_norm=hbvpara_raw-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep,ngage,self.nfea*self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,0,np.newaxis]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]
        hbvpara_delta=torch.clamp(hbvpara0_delta,min=-0.2,max=0.2).view(ntstep,ngage,self.nfea,
                                                                        self.nmul)  # hbv scaled para, [0,1]

        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara=torch.clamp(hbvpara_raw+hbvpara_delta,min=0.0,max=1.0).view(ntstep,ngage,self.nfea,self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out1=self.HBV(x,parameters=hbvpara_raw,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,
                      rtwts=routpara,
                      bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        out2=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                      bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out1,out2


class MultiInv_HBVTDModel_landuse3_twoloss_changeweights_clamp_narrow2(torch.nn.Module):
    # loss_a based on initial para + loss_b on updated para, final loss = loss_a + loss_b*weights
    # need inputing a parameter in lossFun
    # train, crit, script need modified
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_twoloss_changeweights_clamp_narrow2,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1+self.nfea*self.nmul,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw[:self.inittime].size()[0]
            hbvpara_raw[:self.inittime]=hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        for ix in self.tdlst:
            hbvpara_raw[self.inittime:,:,ix-1]=hbvpara_ori[self.inittime:,:,ix-1]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw[:self.inittime,:,ix-1]=hbvpara_ori[self.inittime+self.staind,:,ix-1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_norm=hbvpara_raw-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep,ngage,self.nfea*self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,0,np.newaxis]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]*0.2
        hbvpara_delta=torch.clamp(hbvpara0_delta,min=-0.2,max=0.2).view(ntstep,ngage,self.nfea,
                                                                        self.nmul)  # hbv scaled para, [0,1]

        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara=torch.clamp(hbvpara_raw+hbvpara_delta,min=0.0,max=1.0).view(ntstep,ngage,self.nfea,self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out1=self.HBV(x,parameters=hbvpara_raw,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,
                      rtwts=routpara,
                      bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        out2=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                      bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out1,out2


class MultiInv_HBVTDModel_landuse3_twoloss_changeweights_clamp_narrow_aftermean(torch.nn.Module):
    # loss_a based on initial para + loss_b on updated para, final loss = loss_a + loss_b*weights
    # need inputing a parameter in lossFun
    # train, crit, script need modified
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_twoloss_changeweights_clamp_narrow_aftermean,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        # self.lstminv_landuse=CudnnLstmModel(
        #     nx=1+self.nfea*self.nmul,ny=ntp,hiddenSize=hiddeninv,dr=drinv)
        self.lstminv_landuse=CudnnLstmModel(
            nx=1+self.nfea,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw[:self.inittime].size()[0]
            hbvpara_raw[:self.inittime]=hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        for ix in self.tdlst:
            hbvpara_raw[self.inittime:,:,ix-1]=hbvpara_ori[self.inittime:,:,ix-1]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw[:self.inittime,:,ix-1]=hbvpara_ori[self.inittime+self.staind,:,ix-1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_nocom=hbvpara_raw.median(3)[0]
        hbvpara_raw_nocom_norm=hbvpara_raw_nocom-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        # hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep,ngage,self.nfea*self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_nocom_norm,z[:,:,0,np.newaxis]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]
        hbvpara_delta=torch.clamp(hbvpara0_delta,min=-0.2,max=0.2).view(ntstep,ngage,self.nfea,
                                                                        self.nmul)  # hbv scaled para, [0,1]

        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara=torch.clamp(hbvpara_raw_nocom[:,:,:,None].repeat([1,1,1,self.nmul])+hbvpara_delta,min=0.0,max=1.0).view(
            ntstep,ngage,self.nfea,self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out1=self.HBV(x,parameters=hbvpara_raw,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,
                      rtwts=routpara,
                      bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        out2=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                      bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out1,out2


class MultiInv_HBVTDModel_landuse3_twoloss_changeweights_clamp_narrow_aftermean_mean(torch.nn.Module):
    # loss_a based on initial para + loss_b on updated para, final loss = loss_a + loss_b*weights
    # need inputing a parameter in lossFun
    # train, crit, script need modified
    # class for dPL + HBV with multiple components and some dynamic parameters
    # not use median but use mean of the parameters as the inputs for the 2nd lstm
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_twoloss_changeweights_clamp_narrow_aftermean_mean,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        # self.lstminv_landuse=CudnnLstmModel(
        #     nx=1+self.nfea*self.nmul,ny=ntp,hiddenSize=hiddeninv,dr=drinv)
        self.lstminv_landuse=CudnnLstmModel(
            nx=1+self.nfea,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw[:self.inittime].size()[0]
            hbvpara_raw[:self.inittime]=hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        for ix in self.tdlst:
            hbvpara_raw[self.inittime:,:,ix-1]=hbvpara_ori[self.inittime:,:,ix-1]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw[:self.inittime,:,ix-1]=hbvpara_ori[self.inittime+self.staind,:,ix-1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_nocom=hbvpara_raw.mean(3)
        hbvpara_raw_nocom_norm=hbvpara_raw_nocom-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        # hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep,ngage,self.nfea*self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_nocom_norm,z[:,:,0,np.newaxis]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]
        hbvpara_delta=torch.clamp(hbvpara0_delta,min=-0.2,max=0.2).view(ntstep,ngage,self.nfea,
                                                                        self.nmul)  # hbv scaled para, [0,1]

        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara=torch.clamp(hbvpara_raw_nocom[:,:,:,None].repeat([1,1,1,self.nmul])+hbvpara_delta,min=0.0,max=1.0).view(
            ntstep,ngage,self.nfea,self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out1=self.HBV(x,parameters=hbvpara_raw,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,
                      rtwts=routpara,
                      bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        out2=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                      bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out1,out2


class MultiInv_HBVTDModel_landuse3_twoloss_changeweights_clamp_narrow2_beta(torch.nn.Module):
    # loss_a based on initial para + loss_b on updated para, final loss = loss_a + loss_b*weights, only update beta
    # need inputing a parameter in lossFun
    # train, crit, script need modified
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_twoloss_changeweights_clamp_narrow2_beta,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1+self.nfea*self.nmul,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw[:self.inittime].size()[0]
            hbvpara_raw[:self.inittime]=hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        for ix in self.tdlst:
            hbvpara_raw[self.inittime:,:,ix-1]=hbvpara_ori[self.inittime:,:,ix-1]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw[:self.inittime,:,ix-1]=hbvpara_ori[self.inittime+self.staind,:,ix-1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_norm=hbvpara_raw-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep,ngage,self.nfea*self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,0,np.newaxis]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]*0.2
        hbvpara_delta=torch.clamp(hbvpara0_delta,min=-0.2,max=0.2).view(ntstep,ngage,self.nfea,
                                                                        self.nmul)  # hbv scaled para, [0,1]

        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################
        hbvpara=hbvpara_raw.view(ntstep,ngage,self.nfea,self.nmul)

        hbvpara[:,:,0,:]=torch.clamp(hbvpara_raw+hbvpara_delta,min=0.0,max=1.0).view(ntstep,ngage,self.nfea,self.nmul)[
                         :,:,0,:]
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out1=self.HBV(x,parameters=hbvpara_raw,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,
                      rtwts=routpara,
                      bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        out2=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                      bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out1,out2


class MultiInv_HBVTDModel_landuse3_twoloss_lowslope(torch.nn.Module):
    # loss_a based on initial para + loss_b on updated para, final loss = loss_a + loss_b*2
    # train, crit, script need modified
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_twoloss_lowslope,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1+self.nfea*self.nmul,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw[:self.inittime].size()[0]
            hbvpara_raw[:self.inittime]=hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        for ix in self.tdlst:
            hbvpara_raw[self.inittime:,:,ix-1]=hbvpara_ori[self.inittime:,:,ix-1]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw[:self.inittime,:,ix-1]=hbvpara_ori[self.inittime+self.staind,:,ix-1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_norm=hbvpara_raw-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep,ngage,self.nfea*self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,0,np.newaxis]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]
        hbvpara_delta=custom_sigmoid(hbvpara0_delta).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]

        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara=torch.clamp(hbvpara_raw+hbvpara_delta,min=0.0,max=1.0).view(ntstep,ngage,self.nfea,self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out1=self.HBV(x,parameters=hbvpara_raw,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,
                      rtwts=routpara,
                      bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        out2=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                      bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out1,out2


class MultiInv_HBVTDModel_landuse3_clamp(torch.nn.Module):
    # delta parameter activated by clamp
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_clamp,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1+self.nfea*self.nmul,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw[:self.inittime].size()[0]
            hbvpara_raw[:self.inittime]=hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        for ix in self.tdlst:
            hbvpara_raw[self.inittime:,:,ix-1]=hbvpara_ori[self.inittime:,:,ix-1]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw[:self.inittime,:,ix-1]=hbvpara_ori[self.inittime+self.staind,:,ix-1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_norm=hbvpara_raw-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep,ngage,self.nfea*self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,0,np.newaxis]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]
        hbvpara_delta=torch.clamp(hbvpara0_delta,min=-1.0,max=1.0).view(ntstep,ngage,self.nfea,
                                                                        self.nmul)  # hbv scaled para, [0,1]

        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara=torch.clamp(hbvpara_raw+hbvpara_delta,min=0.0,max=1.0).view(ntstep,ngage,self.nfea,self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_landuse3_clamp_narrow(torch.nn.Module):
    # delta parameter activated by clamp
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_clamp_narrow,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1+self.nfea*self.nmul,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw[:self.inittime].size()[0]
            hbvpara_raw[:self.inittime]=hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        for ix in self.tdlst:
            hbvpara_raw[self.inittime:,:,ix-1]=hbvpara_ori[self.inittime:,:,ix-1]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw[:self.inittime,:,ix-1]=hbvpara_ori[self.inittime+self.staind,:,ix-1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_norm=hbvpara_raw-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep,ngage,self.nfea*self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,0,np.newaxis]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]
        hbvpara_delta=torch.clamp(hbvpara0_delta,min=-0.2,max=0.2).view(ntstep,ngage,self.nfea,
                                                                        self.nmul)  # hbv scaled para, [0,1]

        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara=torch.clamp(hbvpara_raw+hbvpara_delta,min=0.0,max=1.0).view(ntstep,ngage,self.nfea,self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_landuse3_clamp_narrow_onlyrout(torch.nn.Module):
    # delta parameter activated by clamp
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_clamp_narrow_onlyrout,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=ninv,ny=nroutpm,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]

        routpara0=self.lstminv_landuse(z[:,:,:])[-1]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_landuse3_clamp_narrow_coeff(torch.nn.Module):
    # delta parameter activated by clamp
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_clamp_narrow_coeff,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm+2
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1+self.nfea*self.nmul,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        hbvpara_coeff=torch.sigmoid(Params0[:,:,-2:])
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw[:self.inittime].size()[0]
            hbvpara_raw[:self.inittime]=hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        for ix in self.tdlst:
            hbvpara_raw[self.inittime:,:,ix-1]=hbvpara_ori[self.inittime:,:,ix-1]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw[:self.inittime,:,ix-1]=hbvpara_ori[self.inittime+self.staind,:,ix-1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_norm=hbvpara_raw-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep,ngage,self.nfea*self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,0,np.newaxis]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]
        # hbvpara_delta=torch.clamp(hbvpara0_delta,min=-0.2,max=0.2).view(ntstep,ngage,self.nfea,
        #                                                                 self.nmul)  # hbv scaled para, [0,1]

        hbvpara_coeff_max=hbvpara_coeff[:,:,0].unsqueeze(-1).expand_as(hbvpara0_delta)
        hbvpara_coeff_min=hbvpara_coeff[:,:,1].unsqueeze(-1).expand_as(hbvpara0_delta)
        hbvpara_delta=torch.where(hbvpara0_delta>hbvpara_coeff_max,hbvpara_coeff_max,hbvpara0_delta)
        hbvpara_delta=torch.where(hbvpara_delta<hbvpara_coeff_min,hbvpara_coeff_min,hbvpara_delta)
        hbvpara_delta=hbvpara_delta.view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]

        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara=torch.clamp(hbvpara_raw+hbvpara_delta,min=0.0,max=1.0).view(ntstep,ngage,self.nfea,self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,self.nhbvpm+self.nroutpm:self.nhbvpm+self.nroutpm+self.nwtspm]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_landuse3_clamp_narrow2(torch.nn.Module):
    # delta parameter activated by clamp first minimize
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_clamp_narrow2,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1+self.nfea*self.nmul,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw[:self.inittime].size()[0]
            hbvpara_raw[:self.inittime]=hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        for ix in self.tdlst:
            hbvpara_raw[self.inittime:,:,ix-1]=hbvpara_ori[self.inittime:,:,ix-1]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw[:self.inittime,:,ix-1]=hbvpara_ori[self.inittime+self.staind,:,ix-1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_norm=hbvpara_raw-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep,ngage,self.nfea*self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,0,np.newaxis]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]*0.2
        hbvpara_delta=torch.clamp(hbvpara0_delta,min=-0.2,max=0.2).view(ntstep,ngage,self.nfea,
                                                                        self.nmul)  # hbv scaled para, [0,1]

        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara=torch.clamp(hbvpara_raw+hbvpara_delta,min=0.0,max=1.0).view(ntstep,ngage,self.nfea,self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_landuse3_lowslope(torch.nn.Module):
    # delta use custom_sigmoid
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_lowslope,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1+self.nfea*self.nmul,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw[:self.inittime].size()[0]
            hbvpara_raw[:self.inittime]=hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        for ix in self.tdlst:
            hbvpara_raw[self.inittime:,:,ix-1]=hbvpara_ori[self.inittime:,:,ix-1]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw[:self.inittime,:,ix-1]=hbvpara_ori[self.inittime+self.staind,:,ix-1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_norm=hbvpara_raw-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep,ngage,self.nfea*self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,0,np.newaxis]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]
        hbvpara_delta=custom_sigmoid(hbvpara0_delta).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]

        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara=torch.clamp(hbvpara_raw+hbvpara_delta,min=0.0,max=1.0).view(ntstep,ngage,self.nfea,self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_landuse3_lowslope_alpha(torch.nn.Module):
    # delta use custom_sigmoid
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_lowslope_alpha,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1+self.nfea*self.nmul,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw[:self.inittime].size()[0]
            hbvpara_raw[:self.inittime]=hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        for ix in self.tdlst:
            hbvpara_raw[self.inittime:,:,ix-1]=hbvpara_ori[self.inittime:,:,ix-1]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw[:self.inittime,:,ix-1]=hbvpara_ori[self.inittime+self.staind,:,ix-1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_norm=hbvpara_raw-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep,ngage,self.nfea*self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,0,np.newaxis]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]
        hbvpara_delta=custom_sigmoid(hbvpara0_delta).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]

        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara=torch.clamp(hbvpara_raw*hbvpara_delta,min=0.0,max=1.0).view(ntstep,ngage,self.nfea,self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_landuse3_lowslope_alpha_delta(torch.nn.Module):
    # delta use custom_sigmoid
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_lowslope_alpha_delta,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse_alpha=CudnnLstmModel(
            nx=1+self.nfea*self.nmul,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse_delta=CudnnLstmModel(
            nx=1+self.nfea*self.nmul,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw[:self.inittime].size()[0]
            hbvpara_raw[:self.inittime]=hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        for ix in self.tdlst:
            hbvpara_raw[self.inittime:,:,ix-1]=hbvpara_ori[self.inittime:,:,ix-1]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw[:self.inittime,:,ix-1]=hbvpara_ori[self.inittime+self.staind,:,ix-1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_norm=hbvpara_raw-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep,ngage,self.nfea*self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,0,np.newaxis]),dim=2)
        Params0_alpha=self.lstminv_landuse_alpha(z_new)
        Params0_delta=self.lstminv_landuse_delta(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara0_alpha=Params0_alpha[:,:,0:self.nhbvpm]
        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]
        hbvpara_alpha=custom_sigmoid(hbvpara0_alpha).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        hbvpara_delta=custom_sigmoid(hbvpara0_delta).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]

        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara=torch.clamp(hbvpara_raw*hbvpara_alpha+hbvpara_delta,min=0.0,max=1.0).view(ntstep,ngage,self.nfea,
                                                                                          self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_landuse3_scale(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_scale,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1+self.nfea*self.nmul,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw[:self.inittime].size()[0]
            hbvpara_raw[:self.inittime]=hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        for ix in self.tdlst:
            hbvpara_raw[self.inittime:,:,ix-1]=hbvpara_ori[self.inittime:,:,ix-1]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw[:self.inittime,:,ix-1]=hbvpara_ori[self.inittime+self.staind,:,ix-1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_norm=hbvpara_raw-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep,ngage,self.nfea*self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,0,np.newaxis]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]
        hbvpara_delta=torch.tanh(hbvpara0_delta).view(ntstep,ngage,self.nfea,
                                                      self.nmul)*0.1  # hbv scaled para, [-0.5,0.5]

        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara=torch.clamp(hbvpara_raw+hbvpara_delta,min=0.0,max=1.0).view(ntstep,ngage,self.nfea,self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_landuse3_alpha(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_alpha,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1+self.nfea*self.nmul,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw[:self.inittime].size()[0]
            hbvpara_raw[:self.inittime]=hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        for ix in self.tdlst:
            hbvpara_raw[self.inittime:,:,ix-1]=hbvpara_ori[self.inittime:,:,ix-1]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw[:self.inittime,:,ix-1]=hbvpara_ori[self.inittime+self.staind,:,ix-1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_norm=hbvpara_raw-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep,ngage,self.nfea*self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,0,np.newaxis]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]
        hbvpara_delta=torch.tanh(hbvpara0_delta).view(ntstep,ngage,self.nfea,
                                                      self.nmul)*0.5+1  # hbv scaled para, [0.5,1.5]

        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara=torch.clamp(hbvpara_raw*hbvpara_delta,min=0.0,max=1.0).view(ntstep,ngage,self.nfea,self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_landuse3_alpha_delta(torch.nn.Module):
    # new activate custom_sigmoid, extra forcing, more norrow range
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_alpha_delta,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse_alpha=CudnnLstmModel(
            nx=1+self.nfea*self.nmul,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        # self.lstminv_landuse_delta = CudnnLstmModel(
        #     nx=3+1+self.nfea*self.nmul, ny=ntp, hiddenSize=hiddeninv, dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw[:self.inittime].size()[0]
            hbvpara_raw[:self.inittime]=hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        for ix in self.tdlst:
            hbvpara_raw[self.inittime:,:,ix-1]=hbvpara_ori[self.inittime:,:,ix-1]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw[:self.inittime,:,ix-1]=hbvpara_ori[self.inittime+self.staind,:,ix-1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_norm=hbvpara_raw-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep,ngage,self.nfea*self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################
        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,0,np.newaxis]),dim=2)
        # z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,:0+1+3]),dim=2)
        Params0_alpha=self.lstminv_landuse_alpha(z_new)
        # Params0_delta = self.lstminv_landuse_delta(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara0_alpha=Params0_alpha[:,:,0:self.nhbvpm]
        hbvpara_alpha=torch.sigmoid(hbvpara0_alpha).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0.9,1.1]
        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # hbvpara = torch.clamp(hbvpara_alpha, min=0.0, max=1.0).view(ntstep, ngage, self.nfea, self.nmul)
        hbvpara=hbvpara_alpha.view(ntstep,ngage,self.nfea,self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_landuse3_New(torch.nn.Module):
    # class MultiInv_HBVTDModel_landuse3_alpha_delta_beta(torch.nn.Module):
    # new only use land use +initial parameter  output the final parameter, same toch.sigmoid function
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_New,self).__init__()
        # super(MultiInv_HBVTDModel_landuse3_alpha_delta_beta, self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        # self.lstminv_landuse_alpha = CudnnLstmModel(
        #     nx=1+self.nfea*self.nmul, ny=ntp, hiddenSize=hiddeninv, dr=drinv)

        self.lstminv_landuse_alpha=CudnnLstmModel(
            nx=1+self.nfea*self.nmul,ny=self.nmul,hiddenSize=hiddeninv,dr=drinv)

        # self.lstminv_landuse_delta = CudnnLstmModel(
        #     nx=3+1+self.nfea*self.nmul, ny=ntp, hiddenSize=hiddeninv, dr=drinv)

        # self.lstminv_landuse_delta = CudnnLstmModel(
        #     nx=3+1+self.nfea*self.nmul, ny=self.nmul, hiddenSize=hiddeninv, dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw[:self.inittime].size()[0]
            hbvpara_raw[:self.inittime]=hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        for ix in self.tdlst:
            hbvpara_raw[self.inittime:,:,ix-1]=hbvpara_ori[self.inittime:,:,ix-1]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw[:self.inittime,:,ix-1]=hbvpara_ori[self.inittime+self.staind,:,ix-1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_norm=hbvpara_raw-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep,ngage,self.nfea*self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################
        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,0,np.newaxis]),dim=2)
        # z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,:0+1+3]),dim=2)
        Params0_alpha=self.lstminv_landuse_alpha(z_new)
        # Params0_delta = self.lstminv_landuse_delta(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara0_alpha=Params0_alpha[:,:,0:self.nhbvpm]
        hbvpara_alpha=torch.sigmoid(hbvpara0_alpha).view(ntstep,ngage,1,self.nmul)*0.1+1  # hbv scaled para, [0.9,1.1]

        # hbvpara0_delta = Params0_delta[:, :, 0:self.nhbvpm]
        # hbvpara_delta = custom_sigmoid(hbvpara0_delta).view(ntstep, ngage, 1, self.nmul)*0.1 # hbv scaled para, [-0.1,0.1]
        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # hbvpara_beta = torch.clamp(hbvpara_raw[:,:,1,np.newaxis,:]*hbvpara_alpha+hbvpara_delta, min=0.0, max=1.0).view(ntstep, ngage, self.nmul)
        hbvpara_beta=hbvpara_alpha.view(
            ntstep,ngage,self.nmul)
        hbvpara=torch.clone(hbvpara_raw)
        hbvpara[:,:,1,:]=hbvpara_beta
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out


#
#
# class MultiInv_HBVTDModel_landuse3_alpha_delta(torch.nn.Module):
#     # new activate custom_sigmoid, extra forcing, more norrow range
#     # class for dPL + HBV with multiple components and some dynamic parameters
#     def __init__(self, *, ninv, nfea, nmul, hiddeninv, drinv=0.5, inittime=0, routOpt=False, comprout=False,
#                  compwts=False, staind=-1, tdlst=[], dydrop=0.0, ETMod=False):
#         # LSTM Inv + HBV Forward
#         super(MultiInv_HBVTDModel_landuse3_alpha_delta, self).__init__()
#         self.ninv = ninv
#         self.nfea = nfea
#         self.hiddeninv = hiddeninv
#         self.nmul = nmul
#         # get the total number of parameters
#         nhbvpm = nfea*nmul
#         if comprout is False:
#             nroutpm = 2
#         else:
#             nroutpm = nmul*2
#         if compwts is False:
#             nwtspm = 0
#         else:
#             nwtspm = nmul
#         ntp = nhbvpm + nroutpm + nwtspm
#         # ntp = nfea*nmul+nmul+2
#         # ntp = nfea * nmul + 2
#         self.lstminv = CudnnLstmModel(
#             nx=ninv-1, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
#
#         self.lstminv_landuse_alpha = CudnnLstmModel(
#             nx=3+1+self.nfea*self.nmul, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
#
#         self.lstminv_landuse_delta = CudnnLstmModel(
#             nx=3+1+self.nfea*self.nmul, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
#
#         if ETMod is True:
#             # use the added para for ET eq
#             self.HBV = HBVMulTDET()
#         else:
#             # the original HBV para
#             self.HBV = HBVMulTD()
#
#         self.gpu = 1
#         self.inittime=inittime
#         self.routOpt=routOpt
#         self.comprout=comprout
#         self.nhbvpm = nhbvpm
#         self.nwtspm = nwtspm
#         self.nroutpm = nroutpm
#         self.staind = staind
#         self.tdlst = tdlst
#         self.dydrop = dydrop
#
#
#     def forward(self, x, z, doDropMC=False):
#         Params0 = self.lstminv(z[:,:,1:]) # dim: Time, Gage, Para
#
#         ntstep = Params0.shape[0]
#         ngage = Params0.shape[1]
#         # print(Params0)
#         hbvpara0 = Params0[:, :, 0:self.nhbvpm]
#         # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
#         hbvpara_ori = torch.sigmoid(hbvpara0).view(ntstep, ngage, self.nfea, self.nmul) # hbv scaled para, [0,1]
#         ##############################################################################
#         # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
#         hbvpara_raw = torch.clone(hbvpara_ori)
#         Nstep = hbvpara_raw[self.inittime:].size()[0]
#         hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep, 1, 1, 1])
#         if self.inittime > 0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
#             Nstep = hbvpara_raw[:self.inittime].size()[0]
#             hbvpara_raw[:self.inittime] = hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep, 1, 1, 1])
#         for ix in self.tdlst:
#             hbvpara_raw[self.inittime:,:,ix-1] = hbvpara_ori[self.inittime:,:,ix-1]
#             if self.inittime > 0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
#                 hbvpara_raw[:self.inittime, :, ix - 1] = hbvpara_ori[self.inittime+self.staind, :, ix - 1]
#
#         # # Delete specific variables
#         # del hbvpara_ori
#         # # Manually release GPU memory
#         # torch.cuda.empty_cache()
#         ################################################################################
#
#         # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
#         # # Compute the mean and variance along the ngage dimension
#         # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
#         # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
#         hbvpara_raw_mean = torch.tensor([0.5]).cuda()
#         # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
#
#         # Normalize along the ngage dimension
#         # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
#         hbvpara_raw_norm = hbvpara_raw - hbvpara_raw_mean
#
#         # ################################################################################
#         # # Delete specific variables; no gradient computation during the test period
#         # del hbvpara_raw_mean, hbvpara_raw_std
#         # # Manually release GPU memory
#         # torch.cuda.empty_cache()
#         # ################################################################################
#
#         # Transpose matrix dimensions; the default order is 'C'
#         hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep, ngage, self.nfea*self.nmul)
#
#         # ################################################################################
#         # # Delete specific variables
#         # del hbvpara_raw_norm
#         # # Manually release GPU memory
#         # torch.cuda.empty_cache()
#         # ################################################################################
#         # z_new = torch.cat((hbvpara_raw_norm_reshape, z[:, :, 0, np.newaxis]), dim=2)
#         z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,:0+1+3]),dim=2)
#         Params0_alpha = self.lstminv_landuse_alpha(z_new)
#         Params0_delta = self.lstminv_landuse_delta(z_new)
#
#
#         # ################################################################################
#         # # Delete specific variables
#         # del hbvpara_raw_norm_reshape,z_new
#         # # Manually release GPU memory
#         # torch.cuda.empty_cache()
#         # ################################################################################
#
#         hbvpara0_alpha = Params0_alpha[:, :, 0:self.nhbvpm]
#         hbvpara_alpha = custom_sigmoid(hbvpara0_alpha).view(ntstep, ngage, self.nfea, self.nmul)*0.1+1 # hbv scaled para, [0.9,1.1]
#
#         hbvpara0_delta = Params0_delta[:, :, 0:self.nhbvpm]
#         hbvpara_delta = custom_sigmoid(hbvpara0_delta).view(ntstep, ngage, self.nfea, self.nmul)*0.1 # hbv scaled para, [-0.1,0.1]
#         # ################################################################################
#         # # Delete specific variables
#         # del hbvpara0_delta, Params0_delta
#         # # Manually release GPU memory
#         # torch.cuda.empty_cache()
#         # ################################################################################
#
#         hbvpara = torch.clamp(hbvpara_raw*hbvpara_alpha+hbvpara_delta, min=0.0, max=1.0).view(ntstep, ngage, self.nfea, self.nmul)
#         # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)
#
#         routpara0 = Params0[-1, :, self.nhbvpm:self.nhbvpm+self.nroutpm] # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
#         if self.comprout is False:
#             # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
#             routpara = torch.sigmoid(routpara0) # [0,1]
#         else:
#             # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
#             routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2) # [0,1]
#         if self.nwtspm == 0: # simple average for multiple components
#             wts = None
#         else: # weighted average using learned weights
#             wtspara = Params0[-1, :, -self.nwtspm:]
#             wts = F.softmax(wtspara, dim=-1)  # softmax to make sure sum to 1
#
#         # ################################################################################
#         # # Delete specific variables
#         # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
#         # # Manually release GPU memory
#         # torch.cuda.empty_cache()
#         # ################################################################################
#
#         out = self.HBV(x, parameters=hbvpara, staind=self.staind, tdlst=self.tdlst, mu=self.nmul, muwts=wts, rtwts=routpara,
#                           bufftime=self.inittime, routOpt=self.routOpt, comprout=self.comprout, dydrop=self.dydrop)
#         # out = hbvpara.mean(3) # averaging 16 nmul results
#         return out

#
#
# class MultiInv_HBVTDModel_landuse3_alpha_delta_beta(torch.nn.Module):
# # class MultiInv_HBVTDModel_landuse3_alpha_delta_beta(torch.nn.Module):
#     # new activate custom_sigmoid, extra forcing, more norrow range, only beta
#     # class for dPL + HBV with multiple components and some dynamic parameters
#     def __init__(self, *, ninv, nfea, nmul, hiddeninv, drinv=0.5, inittime=0, routOpt=False, comprout=False,
#                  compwts=False, staind=-1, tdlst=[], dydrop=0.0, ETMod=False):
#         # LSTM Inv + HBV Forward
#         super(MultiInv_HBVTDModel_landuse3_New, self).__init__()
#         # super(MultiInv_HBVTDModel_landuse3_alpha_delta_beta, self).__init__()
#         self.ninv = ninv
#         self.nfea = nfea
#         self.hiddeninv = hiddeninv
#         self.nmul = nmul
#         # get the total number of parameters
#         nhbvpm = nfea*nmul
#         if comprout is False:
#             nroutpm = 2
#         else:
#             nroutpm = nmul*2
#         if compwts is False:
#             nwtspm = 0
#         else:
#             nwtspm = nmul
#         ntp = nhbvpm + nroutpm + nwtspm
#         # ntp = nfea*nmul+nmul+2
#         # ntp = nfea * nmul + 2
#         self.lstminv = CudnnLstmModel(
#             nx=ninv-1, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
#
#         # self.lstminv_landuse_alpha = CudnnLstmModel(
#         #     nx=1+self.nfea*self.nmul, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
#
#         self.lstminv_landuse_alpha = CudnnLstmModel(
#             nx=3+1+self.nfea*self.nmul, ny=self.nmul, hiddenSize=hiddeninv, dr=drinv)
#
#         # self.lstminv_landuse_delta = CudnnLstmModel(
#         #     nx=3+1+self.nfea*self.nmul, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
#
#         self.lstminv_landuse_delta = CudnnLstmModel(
#             nx=3+1+self.nfea*self.nmul, ny=self.nmul, hiddenSize=hiddeninv, dr=drinv)
#
#         if ETMod is True:
#             # use the added para for ET eq
#             self.HBV = HBVMulTDET()
#         else:
#             # the original HBV para
#             self.HBV = HBVMulTD()
#
#         self.gpu = 1
#         self.inittime=inittime
#         self.routOpt=routOpt
#         self.comprout=comprout
#         self.nhbvpm = nhbvpm
#         self.nwtspm = nwtspm
#         self.nroutpm = nroutpm
#         self.staind = staind
#         self.tdlst = tdlst
#         self.dydrop = dydrop
#
#
#     def forward(self, x, z, doDropMC=False):
#         Params0 = self.lstminv(z[:,:,1:]) # dim: Time, Gage, Para
#
#         ntstep = Params0.shape[0]
#         ngage = Params0.shape[1]
#         # print(Params0)
#         hbvpara0 = Params0[:, :, 0:self.nhbvpm]
#         # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
#         hbvpara_ori = torch.sigmoid(hbvpara0).view(ntstep, ngage, self.nfea, self.nmul) # hbv scaled para, [0,1]
#         ##############################################################################
#         # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
#         hbvpara_raw = torch.clone(hbvpara_ori)
#         Nstep = hbvpara_raw[self.inittime:].size()[0]
#         hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep, 1, 1, 1])
#         if self.inittime > 0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
#             Nstep = hbvpara_raw[:self.inittime].size()[0]
#             hbvpara_raw[:self.inittime] = hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep, 1, 1, 1])
#         for ix in self.tdlst:
#             hbvpara_raw[self.inittime:,:,ix-1] = hbvpara_ori[self.inittime:,:,ix-1]
#             if self.inittime > 0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
#                 hbvpara_raw[:self.inittime, :, ix - 1] = hbvpara_ori[self.inittime+self.staind, :, ix - 1]
#
#         # # Delete specific variables
#         # del hbvpara_ori
#         # # Manually release GPU memory
#         # torch.cuda.empty_cache()
#         ################################################################################
#
#         # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
#         # # Compute the mean and variance along the ngage dimension
#         # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
#         # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
#         hbvpara_raw_mean = torch.tensor([0.5]).cuda()
#         # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
#
#         # Normalize along the ngage dimension
#         # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
#         hbvpara_raw_norm = hbvpara_raw - hbvpara_raw_mean
#
#         # ################################################################################
#         # # Delete specific variables; no gradient computation during the test period
#         # del hbvpara_raw_mean, hbvpara_raw_std
#         # # Manually release GPU memory
#         # torch.cuda.empty_cache()
#         # ################################################################################
#
#         # Transpose matrix dimensions; the default order is 'C'
#         hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep, ngage, self.nfea*self.nmul)
#
#         # ################################################################################
#         # # Delete specific variables
#         # del hbvpara_raw_norm
#         # # Manually release GPU memory
#         # torch.cuda.empty_cache()
#         # ################################################################################
#         # z_new = torch.cat((hbvpara_raw_norm_reshape, z[:, :, 0, np.newaxis]), dim=2)
#         z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,:0+1+3]),dim=2)
#         Params0_alpha = self.lstminv_landuse_alpha(z_new)
#         Params0_delta = self.lstminv_landuse_delta(z_new)
#
#
#         # ################################################################################
#         # # Delete specific variables
#         # del hbvpara_raw_norm_reshape,z_new
#         # # Manually release GPU memory
#         # torch.cuda.empty_cache()
#         # ################################################################################
#
#         hbvpara0_alpha = Params0_alpha[:, :, 0:self.nhbvpm]
#         hbvpara_alpha = custom_sigmoid(hbvpara0_alpha).view(ntstep, ngage, 1, self.nmul)*0.1+1 # hbv scaled para, [0.9,1.1]
#
#         hbvpara0_delta = Params0_delta[:, :, 0:self.nhbvpm]
#         hbvpara_delta = custom_sigmoid(hbvpara0_delta).view(ntstep, ngage, 1, self.nmul)*0.1 # hbv scaled para, [-0.1,0.1]
#         # ################################################################################
#         # # Delete specific variables
#         # del hbvpara0_delta, Params0_delta
#         # # Manually release GPU memory
#         # torch.cuda.empty_cache()
#         # ################################################################################
#
#         hbvpara_beta = torch.clamp(hbvpara_raw[:,:,1,np.newaxis,:]*hbvpara_alpha+hbvpara_delta, min=0.0, max=1.0).view(ntstep, ngage, self.nmul)
#         hbvpara = torch.clone(hbvpara_raw)
#         hbvpara[:,:,1,:] = hbvpara_beta
#         # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)
#
#         routpara0 = Params0[-1, :, self.nhbvpm:self.nhbvpm+self.nroutpm] # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
#         if self.comprout is False:
#             # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
#             routpara = torch.sigmoid(routpara0) # [0,1]
#         else:
#             # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
#             routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2) # [0,1]
#         if self.nwtspm == 0: # simple average for multiple components
#             wts = None
#         else: # weighted average using learned weights
#             wtspara = Params0[-1, :, -self.nwtspm:]
#             wts = F.softmax(wtspara, dim=-1)  # softmax to make sure sum to 1
#
#         # ################################################################################
#         # # Delete specific variables
#         # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
#         # # Manually release GPU memory
#         # torch.cuda.empty_cache()
#         # ################################################################################
#
#         out = self.HBV(x, parameters=hbvpara, staind=self.staind, tdlst=self.tdlst, mu=self.nmul, muwts=wts, rtwts=routpara,
#                           bufftime=self.inittime, routOpt=self.routOpt, comprout=self.comprout, dydrop=self.dydrop)
#         # out = hbvpara.mean(3) # averaging 16 nmul results
#         return out

#
# class MultiInv_HBVTDModel_landuse3_New(torch.nn.Module):
#     # class for dPL + HBV with multiple components and some dynamic parameters
#     def __init__(self, *, ninv, nfea, nmul, hiddeninv, drinv=0.5, inittime=0, routOpt=False, comprout=False,
#                  compwts=False, staind=-1, tdlst=[], dydrop=0.0, ETMod=False):
#         # LSTM Inv + HBV Forward
#         super(MultiInv_HBVTDModel_landuse3_New, self).__init__()
#         self.ninv = ninv
#         self.nfea = nfea
#         self.hiddeninv = hiddeninv
#         self.nmul = nmul
#         # get the total number of parameters
#         nhbvpm = nfea*nmul
#         if comprout is False:
#             nroutpm = 2
#         else:
#             nroutpm = nmul*2
#         if compwts is False:
#             nwtspm = 0
#         else:
#             nwtspm = nmul
#         ntp = nhbvpm + nroutpm + nwtspm
#         # ntp = nfea*nmul+nmul+2
#         # ntp = nfea * nmul + 2
#         self.lstminv = CudnnLstmModel(
#             nx=ninv-1, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
#
#         self.lstminv_landuse = CudnnLstmModel(
#             nx=1+3+self.nfea*self.nmul, ny=ntp, hiddenSize=hiddeninv, dr=drinv)
#
#         if ETMod is True:
#             # use the added para for ET eq
#             self.HBV = HBVMulTDET()
#         else:
#             # the original HBV para
#             self.HBV = HBVMulTD()
#
#         self.gpu = 1
#         self.inittime=inittime
#         self.routOpt=routOpt
#         self.comprout=comprout
#         self.nhbvpm = nhbvpm
#         self.nwtspm = nwtspm
#         self.nroutpm = nroutpm
#         self.staind = staind
#         self.tdlst = tdlst
#         self.dydrop = dydrop
#
#
#     def forward(self, x, z, doDropMC=False):
#         Params0 = self.lstminv(z[:,:,1:]) # dim: Time, Gage, Para
#
#         ntstep = Params0.shape[0]
#         ngage = Params0.shape[1]
#         # print(Params0)
#         hbvpara0 = Params0[:, :, 0:self.nhbvpm]
#         # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
#         hbvpara_ori = torch.sigmoid(hbvpara0).view(ntstep, ngage, self.nfea, self.nmul) # hbv scaled para, [0,1]
#         ##############################################################################
#         # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
#         hbvpara_raw = torch.clone(hbvpara_ori)
#         Nstep = hbvpara_raw[self.inittime:].size()[0]
#         hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep, 1, 1, 1])
#         if self.inittime > 0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
#             Nstep = hbvpara_raw[:self.inittime].size()[0]
#             hbvpara_raw[:self.inittime] = hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep, 1, 1, 1])
#         for ix in self.tdlst:
#             hbvpara_raw[self.inittime:,:,ix-1] = hbvpara_ori[self.inittime:,:,ix-1]
#             if self.inittime > 0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
#                 hbvpara_raw[:self.inittime, :, ix - 1] = hbvpara_ori[self.inittime+self.staind, :, ix - 1]
#
#         # # Delete specific variables
#         # del hbvpara_ori
#         # # Manually release GPU memory
#         # torch.cuda.empty_cache()
#         ################################################################################
#
#         # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
#         # # Compute the mean and variance along the ngage dimension
#         # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
#         # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
#         hbvpara_raw_mean = torch.tensor([0.5]).cuda()
#         # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
#
#         # Normalize along the ngage dimension
#         # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
#         hbvpara_raw_norm = hbvpara_raw - hbvpara_raw_mean
#
#         # ################################################################################
#         # # Delete specific variables; no gradient computation during the test period
#         # del hbvpara_raw_mean, hbvpara_raw_std
#         # # Manually release GPU memory
#         # torch.cuda.empty_cache()
#         # ################################################################################
#
#         # Transpose matrix dimensions; the default order is 'C'
#         hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep, ngage, self.nfea*self.nmul)
#
#         # ################################################################################
#         # # Delete specific variables
#         # del hbvpara_raw_norm
#         # # Manually release GPU memory
#         # torch.cuda.empty_cache()
#         # ################################################################################
#
#         z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,:0+3+1]),dim=2)
#         Params0_delta = self.lstminv_landuse(z_new)
#
#         # ################################################################################
#         # # Delete specific variables
#         # del hbvpara_raw_norm_reshape,z_new
#         # # Manually release GPU memory
#         # torch.cuda.empty_cache()
#         # ################################################################################
#
#         hbvpara0_delta = Params0_delta[:, :, 0:self.nhbvpm]
#         hbvpara_delta = torch.tanh(hbvpara0_delta).view(ntstep, ngage, self.nfea, self.nmul) # hbv scaled para, [0,1]
#
#         # ################################################################################
#         # # Delete specific variables
#         # del hbvpara0_delta, Params0_delta
#         # # Manually release GPU memory
#         # torch.cuda.empty_cache()
#         # ################################################################################
#
#         hbvpara = torch.clamp(hbvpara_raw+hbvpara_delta, min=0.0, max=1.0).view(ntstep, ngage, self.nfea, self.nmul)
#         # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)
#
#         routpara0 = Params0[-1, :, self.nhbvpm:self.nhbvpm+self.nroutpm] # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
#         if self.comprout is False:
#             # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
#             routpara = torch.sigmoid(routpara0) # [0,1]
#         else:
#             # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
#             routpara = torch.sigmoid(routpara0).view(ngage * self.nmul, 2) # [0,1]
#         if self.nwtspm == 0: # simple average for multiple components
#             wts = None
#         else: # weighted average using learned weights
#             wtspara = Params0[-1, :, -self.nwtspm:]
#             wts = F.softmax(wtspara, dim=-1)  # softmax to make sure sum to 1
#
#         # ################################################################################
#         # # Delete specific variables
#         # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
#         # # Manually release GPU memory
#         # torch.cuda.empty_cache()
#         # ################################################################################
#
#         out = self.HBV(x, parameters=hbvpara, staind=self.staind, tdlst=self.tdlst, mu=self.nmul, muwts=wts, rtwts=routpara,
#                           bufftime=self.inittime, routOpt=self.routOpt, comprout=self.comprout, dydrop=self.dydrop)
#         # out = hbvpara.mean(3) # averaging 16 nmul results
#         return out

class MultiInv_HBVTDModel_landuse3_New_Doaa(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_New_Doaa,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1-3,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=ninv+self.nfea*self.nmul,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1+3:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        # Nstep = hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw=hbvpara_ori[self.staind].unsqueeze(0).repeat([ntstep,1,1,1])

        # if self.inittime > 0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
        #     Nstep = hbvpara_raw[:self.inittime].size()[0]
        #     hbvpara_raw[:self.inittime] = hbvpara_ori[self.inittime + self.staind].unsqueeze(0).repeat([Nstep, 1, 1, 1])
        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_norm=hbvpara_raw-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep,ngage,self.nfea*self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,:]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]
        hbvpara_delta=torch.tanh(hbvpara0_delta).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]

        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara=torch.clamp(hbvpara_raw+hbvpara_delta,min=0.0,max=1.0).view(ntstep,ngage,self.nfea,self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_landuse3_penalty(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_penalty,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1+self.nfea*self.nmul,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw[:self.inittime].size()[0]
            hbvpara_raw[:self.inittime]=hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        for ix in self.tdlst:
            hbvpara_raw[self.inittime:,:,ix-1]=hbvpara_ori[self.inittime:,:,ix-1]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw[:self.inittime,:,ix-1]=hbvpara_ori[self.inittime+self.staind,:,ix-1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_norm=hbvpara_raw-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep,ngage,self.nfea*self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,0,np.newaxis]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]
        hbvpara_delta=torch.tanh(hbvpara0_delta).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]

        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara=torch.clamp(hbvpara_raw+hbvpara_delta,min=0.0,max=1.0).view(ntstep,ngage,self.nfea,self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        # pjli 20240408
        # calculate delta_para_penalty
        delta_para=hbvpara-hbvpara_raw
        delta_loss=0.0
        for ix in range(1,14):
            if ix in self.tdlst:
                delta_loss=delta_loss+torch.mean(torch.pow(delta_para[self.inittime-1:,:,ix-1],2))
            else:
                delta_loss=delta_loss+torch.mean(torch.pow(delta_para[[self.inittime-1,-1],:,ix-1],2))

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out,delta_loss


class MultiInv_HBVTDModel_landuse3_more_free(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_more_free,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1+self.nfea*self.nmul,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_norm=hbvpara_raw-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep,ngage,self.nfea*self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,0,np.newaxis]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]
        hbvpara_delta=torch.tanh(hbvpara0_delta).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]

        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara=torch.clamp(hbvpara_raw+hbvpara_delta,min=0.0,max=1.0).view(ntstep,ngage,self.nfea,self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_landuse3_Dt_limit(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_Dt_limit,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1+self.nfea*self.nmul,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,1:])  # dim: Time, Gage, Para

        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara_ori=torch.sigmoid(hbvpara0).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        ##############################################################################
        # deal with the dynamic parameters and dropout to reduce overfitting of dynamic para
        hbvpara_raw=torch.clone(hbvpara_ori)
        Nstep=hbvpara_raw[self.inittime:].size()[0]
        hbvpara_raw[self.inittime:]=hbvpara_ori[self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
            Nstep=hbvpara_raw[:self.inittime].size()[0]
            hbvpara_raw[:self.inittime]=hbvpara_ori[self.inittime+self.staind].unsqueeze(0).repeat([Nstep,1,1,1])
        ######### pjli 20240329:
        # for ix in self.tdlst:
        for ix in [torch.tensor([1]),torch.tensor([13])]:
            hbvpara_raw[self.inittime:,:,ix-1]=hbvpara_ori[self.inittime:,:,ix-1]
            if self.inittime>0:  # This can cause problems on the test set; if inittime is 0, the dimensions may be incorrect
                hbvpara_raw[:self.inittime,:,ix-1]=hbvpara_ori[self.inittime+self.staind,:,ix-1]

        # # Delete specific variables
        # del hbvpara_ori
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        ################################################################################

        # pjli 20200329, this is the reason why perform poor. The mean and std changes over different data.
        # # Compute the mean and variance along the ngage dimension
        # hbvpara_raw_mean = hbvpara_raw.mean(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])
        hbvpara_raw_mean=torch.tensor([0.5]).cuda()
        # hbvpara_raw_std = hbvpara_raw.std(dim=1, keepdim=True).repeat([1, ngage, 1, 1])

        # Normalize along the ngage dimension
        # hbvpara_raw_norm = (hbvpara_raw - hbvpara_raw_mean) / (hbvpara_raw_std + 1e-8)
        hbvpara_raw_norm=hbvpara_raw-hbvpara_raw_mean

        # ################################################################################
        # # Delete specific variables; no gradient computation during the test period
        # del hbvpara_raw_mean, hbvpara_raw_std
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        # Transpose matrix dimensions; the default order is 'C'
        hbvpara_raw_norm_reshape=hbvpara_raw_norm.view(ntstep,ngage,self.nfea*self.nmul)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        z_new=torch.cat((hbvpara_raw_norm_reshape,z[:,:,0,np.newaxis]),dim=2)
        Params0_delta=self.lstminv_landuse(z_new)

        # ################################################################################
        # # Delete specific variables
        # del hbvpara_raw_norm_reshape,z_new
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]
        hbvpara_delta=torch.tanh(hbvpara0_delta).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]

        # ################################################################################
        # # Delete specific variables
        # del hbvpara0_delta, Params0_delta
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        hbvpara=torch.clamp(hbvpara_raw+hbvpara_delta,min=0.0,max=1.0).view(ntstep,ngage,self.nfea,self.nmul)
        # hbvpara = torch.sigmoid(hbvpara_raw + hbvpara_delta)

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1

        # ################################################################################
        # # Delete specific variables
        # del Params0,hbvpara_raw,hbvpara_delta,hbvpara0,z
        # # Manually release GPU memory
        # torch.cuda.empty_cache()
        # ################################################################################

        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out


class MultiInv_HBVTDModel_landuse2_ori(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse2_ori,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,:-1])  # dim: Time, Gage, Para
        Params0_delta=self.lstminv_landuse(z[:,:,-1].unsqueeze(2))
        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara0_sum=hbvpara0+hbvpara0_delta
        # hbvpara_ori = torch.sigmoid(hbvpara0).view(ntstep, ngage, self.nfea, self.nmul) # hbv scaled para, [0,1]
        hbvpara=torch.sigmoid(hbvpara0_sum).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]
        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        return out


class MultiInv_HBVTDModel_landuse3_ori(torch.nn.Module):
    # class for dPL + HBV with multiple components and some dynamic parameters
    def __init__(self,*,ninv,nfea,nmul,hiddeninv,drinv=0.5,inittime=0,routOpt=False,comprout=False,
                 compwts=False,staind=-1,tdlst=[],dydrop=0.0,ETMod=False):
        # LSTM Inv + HBV Forward
        super(MultiInv_HBVTDModel_landuse3_ori,self).__init__()
        self.ninv=ninv
        self.nfea=nfea
        self.hiddeninv=hiddeninv
        self.nmul=nmul
        # get the total number of parameters
        nhbvpm=nfea*nmul
        if comprout is False:
            nroutpm=2
        else:
            nroutpm=nmul*2
        if compwts is False:
            nwtspm=0
        else:
            nwtspm=nmul
        ntp=nhbvpm+nroutpm+nwtspm
        # ntp = nfea*nmul+nmul+2
        # ntp = nfea * nmul + 2
        self.lstminv=CudnnLstmModel(
            nx=ninv-1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        self.lstminv_landuse=CudnnLstmModel(
            nx=1,ny=ntp,hiddenSize=hiddeninv,dr=drinv)

        if ETMod is True:
            # use the added para for ET eq
            self.HBV=HBVMulTDET()
        else:
            # the original HBV para
            self.HBV=HBVMulTD()

        self.gpu=1
        self.inittime=inittime
        self.routOpt=routOpt
        self.comprout=comprout
        self.nhbvpm=nhbvpm
        self.nwtspm=nwtspm
        self.nroutpm=nroutpm
        self.staind=staind
        self.tdlst=tdlst
        self.dydrop=dydrop

    def forward(self,x,z,doDropMC=False):
        Params0=self.lstminv(z[:,:,:-1])  # dim: Time, Gage, Para
        # z_new = torch.cat((Params0[:, :, 0:self.nhbvpm], z[:, :, -1, np.newaxis]), dim=2)

        Params0_delta=self.lstminv_landuse(z[:,:,-1,np.newaxis])
        ntstep=Params0.shape[0]
        ngage=Params0.shape[1]
        # print(Params0)
        hbvpara0=Params0[:,:,0:self.nhbvpm]
        hbvpara0_delta=Params0_delta[:,:,0:self.nhbvpm]
        # hbvpara = torch.clamp(hbvpara0, min=0.0, max=1.0).view(ngage, self.nfea, self.nmul)
        hbvpara0_sum=hbvpara0+hbvpara0_delta
        # hbvpara_ori = torch.sigmoid(hbvpara0).view(ntstep, ngage, self.nfea, self.nmul) # hbv scaled para, [0,1]
        hbvpara=torch.sigmoid(hbvpara0_sum).view(ntstep,ngage,self.nfea,self.nmul)  # hbv scaled para, [0,1]

        routpara0=Params0[-1,:,self.nhbvpm:self.nhbvpm+self.nroutpm]  # routing para dim:[Ngage, nmul*2] or [Ngage, 2]
        if self.comprout is False:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0)
            routpara=torch.sigmoid(routpara0)  # [0,1]
        else:
            # routpara = torch.clamp(routpara0, min=0.0, max=1.0).view(ngage*self.nmul, 2) # first dim:gage*component
            routpara=torch.sigmoid(routpara0).view(ngage*self.nmul,2)  # [0,1]
        if self.nwtspm==0:  # simple average for multiple components
            wts=None
        else:  # weighted average using learned weights
            wtspara=Params0[-1,:,-self.nwtspm:]
            wts=F.softmax(wtspara,dim=-1)  # softmax to make sure sum to 1
        out=self.HBV(x,parameters=hbvpara,staind=self.staind,tdlst=self.tdlst,mu=self.nmul,muwts=wts,rtwts=routpara,
                     bufftime=self.inittime,routOpt=self.routOpt,comprout=self.comprout,dydrop=self.dydrop)
        # out = hbvpara.mean(3) # averaging 16 nmul results
        return out
