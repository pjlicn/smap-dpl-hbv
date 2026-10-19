import torch
import torch.nn as nn
from torch.nn import Parameter
from ..rnn import CudnnLstm
from hydroDL.model.rnn import UH_gamma,UH_conv
import numpy as np
import torch.nn.functional as F



class CudnnLstmModel_SMAP(torch.nn.Module):
    def __init__(self,*,nx,ny,hiddenSize,dr=0.5):
        super(CudnnLstmModel_SMAP,self).__init__()
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
        out=torch.clamp(self.linearOut(outLSTM),min=0.0, max = 1.0)
        return out

class HBVMulTDET_1_1p_no_g_SMAP(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""

    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first

    def __init__(self,nmul):
        """Initiate a HBV instance"""
        super(HBVMulTDET_1_1p_no_g_SMAP,self).__init__()

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
        SMAP_corr_mu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
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
                SMAP_corr_mu[t,:,:] = SM/parFC

        Qsimave0=Qsimmu0.mean(-1,keepdim=True)
        Qsimave1=Qsimmu1.mean(-1,keepdim=True)
        Qsimave2=Qsimmu2.mean(-1,keepdim=True)
        ETave=ETmu.mean(-1,keepdim=True)
        SWEave=SWEmu.mean(-1,keepdim=True)
        SMAP_corr_ave = SMAP_corr_mu.mean(-1,keepdim=True)

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
            Qall=torch.cat((SMAP_corr_ave,Qsimave0,Qsimave1,Qsimave2,ETave,SWEave),dim=-1)
            return Qall[bufftime_old:]

    class HBVMulTDET_1_1p_no_g_SMAP(torch.nn.Module):
        """HBV Model with multiple components and dynamic parameters PyTorch version"""

        # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
        # we suggest you read the class HBVMul() with original static parameters first

        def __init__(self,nmul):
            """Initiate a HBV instance"""
            super(HBVMulTDET_1_1p_no_g_SMAP,self).__init__()

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
                drmask=torch.bernoulli(pmat).detach_().to(
                    parhbvFull)  # to drop dynamic parameters as static in some basins
                comPar=dynPar*(1-drmask)+staPar*drmask
                parhbvFull[:,:,ix-1,:]=comPar

            # Initialize time series of model variables to save results
            Qsimmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
            ETmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
            SWEmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
            SMAP_corr_mu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
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
                    SMAP_corr_mu[t,:,:]=SM/parFC

            Qsimave0=Qsimmu0.mean(-1,keepdim=True)
            Qsimave1=Qsimmu1.mean(-1,keepdim=True)
            Qsimave2=Qsimmu2.mean(-1,keepdim=True)
            ETave=ETmu.mean(-1,keepdim=True)
            SWEave=SWEmu.mean(-1,keepdim=True)
            SMAP_corr_ave=SMAP_corr_mu.mean(-1,keepdim=True)

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
                Qall=torch.cat((SMAP_corr_ave,Qsimave0,Qsimave1,Qsimave2,ETave,SWEave),dim=-1)
                return Qall[bufftime_old:]

class HBVMulTDET_1_1p_no_g_SMAP_ab(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""

    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first

    def __init__(self,nmul):
        """Initiate a HBV instance"""
        super(HBVMulTDET_1_1p_no_g_SMAP_ab,self).__init__()

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
        hbvscaLst=[[1,6],[50,1000],[0.05,0.9],[0.01,0.5],[0.001,0.2],[0.2,1],
                   # [0,10],[0,100],[-2.5,2.5],[0.5,10],[0,0.1],[0,0.2],[0.3,5],[0,1],[2,60],[-1,1]]  # HBV para
                   [0,10],[0,100],[-2.5,2.5],[0.5,10],[0,0.1],[0,0.2],[0.3,5],[0,1],[0,5],[-1,1]]  # HBV para
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
        SMAP_corr_mu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
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

                parBETA,parFC,parK0,parK1,parK2,parLP,parPERC,parUZL,parTT,parCFMAX,parCFR,parCWH,parBETAET,parC,parA,parB=paraLst
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
                # SMAP_corr_mu[t,:,:]=(SM/parFC)
                SMAP_corr_mu[t,:,:] = torch.clamp((SM/parFC-parB)*parA,min=0.0, max=1.0)

        Qsimave0=Qsimmu0.mean(-1,keepdim=True)
        Qsimave1=Qsimmu1.mean(-1,keepdim=True)
        Qsimave2=Qsimmu2.mean(-1,keepdim=True)
        ETave=ETmu.mean(-1,keepdim=True)
        SWEave=SWEmu.mean(-1,keepdim=True)
        SMAP_corr_ave = SMAP_corr_mu.mean(-1,keepdim=True)
        # SMAP_ave_time = SMAP_corr_ave.mean(0,keepdim=True)
        # parA_ave = parA.mean(1,keepdim=True)[:,:,None]
        # parB_ave = parB.mean(1,keepdim=True)[:,:,None]
        # SMAP_trans = SMAP_corr_ave*parA_ave + (1-parA_ave)*SMAP_ave_time + parB_ave

        # # get the initial average
        # if muwts is None:  # simple average
        #     Qsimave=Qsimmu.mean(-1)
        # else:  # weighted average using learned weights
        #     Qsimave=(Qsimmu*muwts).sum(-1)
        #
        # if routOpt is True:  # routing
        #     if comprout is True:
        #         # do routing to all the components, reshape the mat to [Time, gage*multi]
        #         Qsim=Qsimmu.view(Nstep,Ngrid*mu)
        #     else:
        #         # average the components, then do routing
        #         Qsim=Qsimave
        #
        #     # scale two routing parameters
        #     tempa=routscaLst[0][0]+rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
        #     tempb=routscaLst[1][0]+rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])
        #     routa=tempa.repeat(Nstep,1).unsqueeze(-1)
        #     routb=tempb.repeat(Nstep,1).unsqueeze(-1)
        #     UH=UH_gamma(routa,routb,lenF=15)  # lenF: folter
        #     rf=torch.unsqueeze(Qsim,-1).permute([1,2,0])  # dim:gage*var*time
        #     UH=UH.permute([1,2,0])  # dim: gage*var*time
        #     Qsrout=UH_conv(rf,UH).permute([2,0,1])
        #
        #     if comprout is True:  # Qs is [time, [gage*mult], var] now
        #         Qstemp=Qsrout.view(Nstep,Ngrid,mu)
        #         if muwts is None:
        #             Qs=Qstemp.mean(-1,keepdim=True)
        #         else:
        #             Qs=(Qstemp*muwts).sum(-1,keepdim=True)
        #     else:
        #         Qs=Qsrout
        #
        # else:  # no routing, output the initial average simulations
        #     Qs=torch.unsqueeze(Qsimave,-1)  # add a dimension

        if outstate is True:  # output states
            # return Qs,SNOWPACK,MELTWATER,SM,SUZ,SLZ
            return SNOWPACK,MELTWATER,SM,SUZ,SLZ
        else:
            # return Qs
            Qall=torch.cat((SMAP_corr_ave,Qsimave0,Qsimave1,Qsimave2,ETave,SWEave),dim=-1)
            return Qall[bufftime_old:]

class HBVMulTDET_1_1p_no_g_surrogate_SMAP(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""

    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first

    def __init__(self,surrogate,surrogate_stat):
        """Initiate a HBV instance"""
        super(HBVMulTDET_1_1p_no_g_surrogate_SMAP,self).__init__()
        self.surrogate = surrogate # parameters are fixed
        self.surrogate_stat = surrogate_stat
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

        # Without warm-up bufftime=0, initialize state variables with zeros
        Ngrid=x.shape[1]

        P=x[bufftime:,:,0]
        Pm=P.unsqueeze(2).repeat(1,1,mu)  # precip
        T=x[bufftime:,:,1]
        Tm=T.unsqueeze(2).repeat(1,1,mu)  # temperature
        ETpot=x[bufftime:,:,2]
        ETpm=ETpot.unsqueeze(2).repeat(1,1,mu)  # potential ET
        parAll=parameters[bufftime:,:,:,:]
        parAllTrans=torch.zeros_like(parAll)

        ## scale the parameters to real values from [0,1]
        hbvscaLst=[[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],
                   [0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]]# ,[0,5],[-1,1]]  # HBV para

        for ip in range(len(hbvscaLst)):  # not include routing. Scaling the parameters
            parAllTrans[:,:,ip,:]=hbvscaLst[ip][0]+parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        # parAllTrans = parAll

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

        # target variable
        SMAP_corr_mu_raw=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)

        # parameter for all 16 realizations
        surrogate_par = parhbvFull[:,:,:,:]

        for i in range(mu):
            # grid, time, feature
            surrogate_input = torch.cat((x,surrogate_par[:,:,:,i]),dim=-1)
            surrogate_input_norm = (surrogate_input-self.surrogate_stat[:,:,:17,0])/self.surrogate_stat[:,:,:17,1]
            # grid, time
            surrogate_output = self.surrogate(surrogate_input_norm.permute(1,0,2)).permute(1,0,2).squeeze()
            surrogate_output_denorm = surrogate_output * self.surrogate_stat[0,0,17,1] + self.surrogate_stat[0,0,17,0]
            SMAP_corr_mu_raw[:,:,i] = torch.clamp(surrogate_output_denorm,min=0.0,max=1.0)

        SMAP_corr_mu = torch.clamp(SMAP_corr_mu_raw,min=0.0, max=1.0)
        # SMAP_corr_mu=torch.clamp((SMAP_corr_mu_raw-parB)*parA,min=0.0,max=1.0)
        SMAP_corr_ave = SMAP_corr_mu.mean(-1,keepdim=True)

        return SMAP_corr_ave[bufftime_old:]


class HBVMulTDET_1_1p_no_g_surrogate_SMAP_ab(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""

    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first

    def __init__(self,surrogate,surrogate_stat):
        """Initiate a HBV instance"""
        super(HBVMulTDET_1_1p_no_g_surrogate_SMAP_ab,self).__init__()
        self.surrogate = surrogate # parameters are fixed
        self.surrogate_stat = surrogate_stat
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

        # Without warm-up bufftime=0, initialize state variables with zeros
        Ngrid=x.shape[1]

        P=x[bufftime:,:,0]
        Pm=P.unsqueeze(2).repeat(1,1,mu)  # precip
        T=x[bufftime:,:,1]
        Tm=T.unsqueeze(2).repeat(1,1,mu)  # temperature
        ETpot=x[bufftime:,:,2]
        ETpm=ETpot.unsqueeze(2).repeat(1,1,mu)  # potential ET
        parAll=parameters[bufftime:,:,:,:]
        parAllTrans=torch.zeros_like(parAll)

        ## scale the parameters to real values from [0,1]
        hbvscaLst=[[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],
                   [0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,5],[-1,1]]  # HBV para

        for ip in range(len(hbvscaLst)):  # not include routing. Scaling the parameters
            parAllTrans[:,:,ip,:]=hbvscaLst[ip][0]+parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        # parAllTrans = parAll

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

        # target variable
        SMAP_corr_mu_raw=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)

        # parameter for all 16 realizations
        surrogate_par = parhbvFull[:,:,:-2,:]
        parA = parhbvFull[:,:,-2,:]
        parB = parhbvFull[:,:,-1,:]

        for i in range(mu):
            # grid, time, feature
            surrogate_input = torch.cat((x,surrogate_par[:,:,:,i]),dim=-1)
            surrogate_input_norm = (surrogate_input-self.surrogate_stat[:,:,:17,0])/self.surrogate_stat[:,:,:17,1]
            # grid, time
            surrogate_output = self.surrogate(surrogate_input_norm.permute(1,0,2)).permute(1,0,2).squeeze()
            surrogate_output_denorm = surrogate_output * self.surrogate_stat[0,0,17,1] + self.surrogate_stat[0,0,17,0]
            SMAP_corr_mu_raw[:,:,i] = torch.clamp(surrogate_output_denorm,min=0.0,max=1.0)

        SMAP_corr_mu = torch.clamp((SMAP_corr_mu_raw-parB)*parA,min=0.0, max=1.0)
        SMAP_corr_ave = SMAP_corr_mu.mean(-1,keepdim=True)

        return SMAP_corr_ave[bufftime_old:]

class HBVMulTDET_1_1p_no_g_surrogate_SMAP_ab_v2(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""

    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first

    def __init__(self,surrogate,surrogate_stat):
        """Initiate a HBV instance"""
        super(HBVMulTDET_1_1p_no_g_surrogate_SMAP_ab_v2,self).__init__()
        self.surrogate = surrogate # parameters are fixed
        self.surrogate_stat = surrogate_stat
    def forward(self,x,parameters,staind,tdlst,mu,muwts,rtwts,bufftime=0,outstate=False,routOpt=False,
                comprout=False,dydrop=False,range_a = [0,5], range_b=[-1,1]):
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

        # Without warm-up bufftime=0, initialize state variables with zeros
        Ngrid=x.shape[1]

        P=x[bufftime:,:,0]
        Pm=P.unsqueeze(2).repeat(1,1,mu)  # precip
        T=x[bufftime:,:,1]
        Tm=T.unsqueeze(2).repeat(1,1,mu)  # temperature
        ETpot=x[bufftime:,:,2]
        ETpm=ETpot.unsqueeze(2).repeat(1,1,mu)  # potential ET
        parAll=parameters[bufftime:,:,:,:]
        parAllTrans=torch.zeros_like(parAll)

        ## scale the parameters to real values from [0,1]
        hbvscaLst=[[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],
                   [0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]]  # HBV para
        hbvscaLst.extend([range_a,range_b])

        for ip in range(len(hbvscaLst)):  # not include routing. Scaling the parameters
            parAllTrans[:,:,ip,:]=hbvscaLst[ip][0]+parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        # parAllTrans = parAll

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

        # target variable
        SMAP_corr_mu_raw=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)

        # parameter for all 16 realizations
        surrogate_par = parhbvFull[:,:,:-2,:]
        parA = parhbvFull[:,:,-2,:]
        parB = parhbvFull[:,:,-1,:]

        for i in range(mu):
            # grid, time, feature
            x_norm = (x -self.surrogate_stat[:,:,:3,0])/self.surrogate_stat[:,:,:3,1]
            surrogate_input_norm = torch.cat((x_norm,surrogate_par[:,:,:,i]),dim=-1)
            # grid, time
            # surrogate_output = self.surrogate(surrogate_input_norm.permute(1,0,2)).permute(1,0,2).squeeze()
            surrogate_output=self.surrogate(surrogate_input_norm.permute(0,1,2)).permute(0,1,2).squeeze()
            SMAP_corr_mu_raw[:,:,i] = torch.clamp(surrogate_output,min=0.0,max=1.0)

        SMAP_corr_mu = torch.clamp((SMAP_corr_mu_raw-parB)*parA,min=0.0, max=1.0)
        SMAP_corr_ave = SMAP_corr_mu.mean(-1,keepdim=True)

        return SMAP_corr_ave[bufftime_old:]

class HBVMulTDET_1_1p_no_g_surrogate_SMAP_ab_v3(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""

    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first
    # surrogate model is ANN

    def __init__(self,surrogate,surrogate_stat):
        """Initiate a HBV instance"""
        super(HBVMulTDET_1_1p_no_g_surrogate_SMAP_ab_v3,self).__init__()
        self.surrogate = surrogate # parameters are fixed
        self.surrogate_stat = surrogate_stat
    def forward(self,x,parameters,staind,tdlst,mu,muwts,rtwts,bufftime=0,outstate=False,routOpt=False,
                comprout=False,dydrop=False,range_a = [0,5], range_b=[-1,1]):
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

        # Without warm-up bufftime=0, initialize state variables with zeros
        Ngrid=x.shape[1]

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

        hbvscaLst.extend([range_a,range_b])

        for ip in range(len(hbvscaLst)):  # not include routing. Scaling the parameters
            parAllTrans[:,:,ip,:]=hbvscaLst[ip][0]+parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        # parAllTrans = parAll

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

        # target variable
        SMAP_corr_mu_raw=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)

        # parameter for all 16 realizations
        surrogate_par = parhbvFull[:,:,:-2,:]
        parA = parhbvFull[:,:,-2,:]
        parB = parhbvFull[:,:,-1,:]


        # Without warm-up bufftime=0, initialize state variables with zeros
        Ngrid=x.shape[1]
        SNOWPACK=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        MELTWATER=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SM=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SUZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)
        SLZ=(torch.zeros([Ngrid,mu],dtype=torch.float32)+0.001).to(x)

        P=x[bufftime:,:,0]
        Pm=P.unsqueeze(2).repeat(1,1,mu)  # precip
        T=x[bufftime:,:,1]
        Tm=T.unsqueeze(2).repeat(1,1,mu)  # temperature
        ETpot=x[bufftime:,:,2]
        ETpm=ETpot.unsqueeze(2).repeat(1,1,mu)  # potential ET


        for t in range(Nstep):
            with torch.no_grad() if t<bufftime_old else torch.enable_grad():
                paraLst=[]

                for ip in range(len(hbvscaLst)-2):  # unpack HBV parameters
                    paraLst.append(parhbvFull[t,:,ip,:])

                # parBETA,parFC,parK0,parK1,parK2,parLP,parPERC,parUZL,parTT,parCFMAX,parCFR,parCWH,parBETAET,parC=paraLst

                # # water use parameters
                # tempc_all=tempc1+tempc2+tempc3
                # routc1=(tempc1/(tempc_all)).mean(-1).unsqueeze(-1)
                # routc2=(tempc2/(tempc_all)).mean(-1).unsqueeze(-1)
                # routc3=(tempc3/(tempc_all)).mean(-1).unsqueeze(-1)

                # Separate precipitation into liquid and solid components
                PRECIP=Pm[t,:,:]
                TEMP=Tm[t,:,:]
                EVAP_P=ETpm[t,:,:]

                for mm in range(mu):
                    surrogate_input = torch.cat((PRECIP[:,mm,None],TEMP[:,mm,None],EVAP_P[:,mm,None],parhbvFull[t,:,:14,mm],SNOWPACK[:,mm,None],MELTWATER[:,mm,None],SM[:,mm,None],SUZ[:,mm,None],SLZ[:,mm,None]),dim=1)
                    surrogate_input_norm=(surrogate_input-self.surrogate_stat[:-5,0][None])/self.surrogate_stat[:-5,1][None]
                    surrogate_output_norm=self.surrogate(surrogate_input_norm)
                    # denorm
                    surrogate_output = surrogate_output_norm * self.surrogate_stat[-5:,1][None] + self.surrogate_stat[-5:,0][None]
                    SNOWPACK[:,mm] = surrogate_output[:,0]
                    MELTWATER[:,mm] = surrogate_output[:,1]
                    SM[:,mm] = surrogate_output[:,2]
                    SUZ[:,mm] = surrogate_output[:,3]
                    SLZ[:,mm] = surrogate_output[:,4]

                # Use torch.where to update conditionally
                condition=torch.tensor(t>=bufftime_old,device=SM.device).reshape(1,1,1)
                temp_result=SM/(parhbvFull[t,:,1,:]+1e-6)
                temp_result=torch.clamp(temp_result,min=0.0,max=1.0)
                SMAP_corr_mu_raw=torch.where(condition,temp_result,temp_result.clone())

        SMAP_corr_mu = torch.clamp((SMAP_corr_mu_raw-parB)*parA,min=0.0, max=1.0)
        SMAP_corr_ave = SMAP_corr_mu.mean(-1,keepdim=True)

        return SMAP_corr_ave[bufftime_old:]

class HBVMulTDET_1_1p_no_g_SMAP_ab_para(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""

    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first

    def __init__(self,nmul):
        """Initiate a HBV instance"""
        super(HBVMulTDET_1_1p_no_g_SMAP_ab_para,self).__init__()

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
        hbvscaLst=[[1,6],[50,1000],[0.05,0.9],[0.01,0.5],[0.001,0.2],[0.2,1],
                   # [0,10],[0,100],[-2.5,2.5],[0.5,10],[0,0.1],[0,0.2],[0.3,5],[0,1],[2,60],[-1,1]]  # HBV para
                   [0,10],[0,100],[-2.5,2.5],[0.5,10],[0,0.1],[0,0.2],[0.3,5],[0,1],[0,5],[-1,1]]  # HBV para
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

        # # Initialize time series of model variables to save results
        # Qsimmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        # ETmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        # SWEmu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        # SMAP_corr_mu=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        # # Output the box components of Q
        # Qsimmu0=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        # Qsimmu1=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)
        # Qsimmu2=(torch.zeros(Pm.size(),dtype=torch.float32)+0.001).to(parhbvFull)

        # # Not used. Logging the state variables for debug.
        # # SMlog = np.zeros(P.size())
        # logSM = np.zeros(P.size())
        # logPS = np.zeros(P.size())
        # logswet = np.zeros(P.size())
        # logRE = np.zeros(P.size())

        # for t in range(Nstep):
        #     with torch.no_grad() if t<bufftime_old else torch.enable_grad():
        #         paraLst=[]
        #         for ip in range(len(hbvscaLst)):  # unpack HBV parameters
        #             paraLst.append(parhbvFull[t,:,ip,:])

                # parBETA,parFC,parK0,parK1,parK2,parLP,parPERC,parUZL,parTT,parCFMAX,parCFR,parCWH,parBETAET,parC,parA,parB=paraLst
                # # Separate precipitation into liquid and solid components
                # PRECIP=Pm[t,:,:]
                # TEMP=Tm[t,:,:]
                # EVAP_P=ETpm[t,:,:]
                #
                # RAIN=torch.mul(PRECIP,(TEMP>=parTT).type(torch.float32))
                # SNOW=torch.mul(PRECIP,(TEMP<parTT).type(torch.float32))
                #
                # # Snow process
                # SNOWPACK=SNOWPACK+SNOW
                # melt=parCFMAX*(TEMP-parTT)
                # melt=torch.clamp(melt,min=0.0)
                # melt=torch.min(melt,SNOWPACK)
                # MELTWATER=MELTWATER+melt
                # SNOWPACK=SNOWPACK-melt
                # refreezing=parCFR*parCFMAX*(parTT-TEMP)
                # refreezing=torch.clamp(refreezing,min=0.0)
                # refreezing=torch.min(refreezing,MELTWATER)
                # SNOWPACK=SNOWPACK+refreezing
                # MELTWATER=MELTWATER-refreezing
                # tosoil=MELTWATER-(parCWH*SNOWPACK)
                # tosoil=torch.clamp(tosoil,min=0.0)
                # MELTWATER=MELTWATER-tosoil
                #
                # # Soil and evaporation
                # soil_wetness=(SM/parFC)**parBETA
                # soil_wetness=torch.clamp(soil_wetness,min=0.0,max=1.0)
                # recharge=(RAIN+tosoil)*soil_wetness
                #
                # # Not used, logging states for checking
                # # logSM[t,:] = SM.detach().cpu().numpy()
                # # logPS[t,:] = (RAIN + tosoil).detach().cpu().numpy()
                # # logswet[t,:] = (SM / parFC).detach().cpu().numpy()
                # # logRE[t, :] = recharge.detach().cpu().numpy()
                #
                # SM=SM+RAIN+tosoil-recharge
                # excess=SM-parFC
                # excess=torch.clamp(excess,min=0.0)
                # SM=SM-excess
                # # MODIFY here. Different as class HBVMulT(). Add a ET shape parameter parBETAET
                # evapfactor=(SM/(parLP*parFC))**parBETAET
                # evapfactor=torch.clamp(evapfactor,min=0.0,max=1.0)
                # ETact=EVAP_P*evapfactor
                # ETact=torch.min(SM,ETact)
                # SM=torch.clamp(SM-ETact,min=PRECS)  # SM can not be zero for gradient tracking
                #
                # capillary=torch.min(SLZ,parC*SLZ*(1.0-torch.clamp(SM/parFC,max=1.0)))
                #
                # SM=torch.clamp(SM+capillary,min=PRECS)
                # SLZ=torch.clamp(SLZ-capillary,min=PRECS)
                #
                # # Groundwater boxes
                # SUZ=SUZ+recharge+excess
                # PERC=torch.min(SUZ,parPERC)
                # SUZ=SUZ-PERC
                # Q0=parK0*torch.clamp(SUZ-parUZL,min=0.0)
                # SUZ=SUZ-Q0
                # Q1=parK1*SUZ
                # SUZ=SUZ-Q1
                # SLZ=SLZ+PERC
                # Q2=parK2*SLZ
                # SLZ=SLZ-Q2
                # Qsimmu[t,:,:]=Q0+Q1+Q2
                #
                # # save components for Q
                # Qsimmu0[t,:,:]=Q0
                # Qsimmu1[t,:,:]=Q1
                # Qsimmu2[t,:,:]=Q2
                # ETmu[t,:,:]=ETact
                # SWEmu[t,:,:]=SNOWPACK
                # # Not used, for debug state variables
                # # SMlog[t,:, :] = SM.detach().cpu().numpy()
                # # SUZlog[t,:,:] = SUZ.detach().cpu().numpy()
                # # SLZlog[t,:,:] = SLZ.detach().cpu().numpy()
                # # SMAP_corr_mu[t,:,:]=(SM/parFC)
                # SMAP_corr_mu[t,:,:] = torch.clamp((SM/parFC-parB)*parA,min=0.0, max=1.0)

        # Qsimave0=Qsimmu0.mean(-1,keepdim=True)
        # Qsimave1=Qsimmu1.mean(-1,keepdim=True)
        # Qsimave2=Qsimmu2.mean(-1,keepdim=True)
        # ETave=ETmu.mean(-1,keepdim=True)
        # SWEave=SWEmu.mean(-1,keepdim=True)
        # SMAP_corr_ave = SMAP_corr_mu.mean(-1,keepdim=True)
        # SMAP_ave_time = SMAP_corr_ave.mean(0,keepdim=True)
        # parA_ave = parA.mean(1,keepdim=True)[:,:,None]
        # parB_ave = parB.mean(1,keepdim=True)[:,:,None]
        # SMAP_trans = SMAP_corr_ave*parA_ave + (1-parA_ave)*SMAP_ave_time + parB_ave

        # # get the initial average
        # if muwts is None:  # simple average
        #     Qsimave=Qsimmu.mean(-1)
        # else:  # weighted average using learned weights
        #     Qsimave=(Qsimmu*muwts).sum(-1)
        #
        # if routOpt is True:  # routing
        #     if comprout is True:
        #         # do routing to all the components, reshape the mat to [Time, gage*multi]
        #         Qsim=Qsimmu.view(Nstep,Ngrid*mu)
        #     else:
        #         # average the components, then do routing
        #         Qsim=Qsimave
        #
        #     # scale two routing parameters
        #     tempa=routscaLst[0][0]+rtwts[:,0]*(routscaLst[0][1]-routscaLst[0][0])
        #     tempb=routscaLst[1][0]+rtwts[:,1]*(routscaLst[1][1]-routscaLst[1][0])
        #     routa=tempa.repeat(Nstep,1).unsqueeze(-1)
        #     routb=tempb.repeat(Nstep,1).unsqueeze(-1)
        #     UH=UH_gamma(routa,routb,lenF=15)  # lenF: folter
        #     rf=torch.unsqueeze(Qsim,-1).permute([1,2,0])  # dim:gage*var*time
        #     UH=UH.permute([1,2,0])  # dim: gage*var*time
        #     Qsrout=UH_conv(rf,UH).permute([2,0,1])
        #
        #     if comprout is True:  # Qs is [time, [gage*mult], var] now
        #         Qstemp=Qsrout.view(Nstep,Ngrid,mu)
        #         if muwts is None:
        #             Qs=Qstemp.mean(-1,keepdim=True)
        #         else:
        #             Qs=(Qstemp*muwts).sum(-1,keepdim=True)
        #     else:
        #         Qs=Qsrout
        #
        # else:  # no routing, output the initial average simulations
        #     Qs=torch.unsqueeze(Qsimave,-1)  # add a dimension

        if outstate is True:  # output states
            # return Qs,SNOWPACK,MELTWATER,SM,SUZ,SLZ
            return SNOWPACK,MELTWATER,SM,SUZ,SLZ
        else:
            # return Qs
            # Qall=torch.cat((SMAP_corr_ave,Qsimave0,Qsimave1,Qsimave2,ETave,SWEave),dim=-1)
            # return Qall[bufftime_old:]
            # paraLst_array=parhbvFull.mean(-1).squeeze()
            paraLst_array=parhbvFull.flatten(start_dim=-2)
            return paraLst_array[bufftime_old:]

class HBVMulTDET_1_1p_no_g_surrogate_SMAP_ab_para(torch.nn.Module):
    """HBV Model with multiple components and dynamic parameters PyTorch version"""

    # Add an ET shape parameter for the original ET equation; others are the same as HBVMulTD()
    # we suggest you read the class HBVMul() with original static parameters first

    def __init__(self,surrogate,surrogate_stat):
        """Initiate a HBV instance"""
        super(HBVMulTDET_1_1p_no_g_surrogate_SMAP_ab_para,self).__init__()
        self.surrogate = surrogate # parameters are fixed
        self.surrogate_stat = surrogate_stat
    def forward(self,x,parameters,staind,tdlst,mu,muwts,rtwts,bufftime=0,outstate=False,routOpt=False,
                comprout=False,dydrop=False,range_a = [0,5], range_b=[-1,1]):
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

        # Without warm-up bufftime=0, initialize state variables with zeros
        Ngrid=x.shape[1]

        P=x[bufftime:,:,0]
        Pm=P.unsqueeze(2).repeat(1,1,mu)  # precip
        T=x[bufftime:,:,1]
        Tm=T.unsqueeze(2).repeat(1,1,mu)  # temperature
        ETpot=x[bufftime:,:,2]
        ETpm=ETpot.unsqueeze(2).repeat(1,1,mu)  # potential ET
        parAll=parameters[bufftime:,:,:,:]
        parAllTrans=torch.zeros_like(parAll)

        ## scale the parameters to real values from [0,1]
        hbvscaLst=[[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],
                   [0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]]  # HBV para
        hbvscaLst+=[range_a,range_b]

        for ip in range(len(hbvscaLst)):  # not include routing. Scaling the parameters
            parAllTrans[:,:,ip,:]=hbvscaLst[ip][0]+parAll[:,:,ip,:]*(hbvscaLst[ip][1]-hbvscaLst[ip][0])

        # parAllTrans = parAll

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

        paraLst_array=parhbvFull.flatten(start_dim=-2)
        return paraLst_array[bufftime_old:]