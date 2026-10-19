import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))

from hydroDL.model import rnn, crit, train
from hydroDL.post import stat
# from hydroDL.master import loadModel
from hydroDL.master import loadStateDict
import torch.nn.functional as F
import argparse
import os
import numpy as np
import torch
from collections import OrderedDict
import random
import json
import pandas as pd
import json
import datetime as dt
import yaml


def parse_args():
    default_config = Path(__file__).resolve().parents[1] / 'configs' / 'example_config.yaml'
    parser = argparse.ArgumentParser(
        description='Portable testing entry point for the SMAP dPL-HBV reference package.'
    )
    parser.add_argument('--config', default=str(default_config), help='YAML configuration file.')
    parser.add_argument('--dataset-npz', help='Combined NPZ with dynamic data, static data, names, and selected IDs.')
    parser.add_argument('--pet-csv', help='PEVAP CSV aligned to the source grid and experiment dates.')
    parser.add_argument('--model-dir', help='Experiment directory containing state_dict and scaler JSON.')
    parser.add_argument('--output-dir', help='Directory for out0-out4 and evaluation products.')
    parser.add_argument('--epoch', type=int, default=100, help='Checkpoint epoch in model-dir.')
    parser.add_argument('--gpu', type=int, default=0, help='CUDA device index; ignored without CUDA.')
    parser.add_argument('--seed', type=int, default=111111, help='Random seed.')
    known, _ = parser.parse_known_args()
    config_path = Path(known.config)
    if config_path.exists():
        with config_path.open(encoding='utf-8') as stream:
            config = yaml.safe_load(stream) or {}
        parser.set_defaults(**config.get('test', {}))
    args = parser.parse_args()
    missing = [
        name for name in ('dataset_npz', 'pet_csv', 'model_dir', 'output_dir')
        if not getattr(args, name)
    ]
    if missing:
        parser.error('missing configuration values: ' + ', '.join(missing))
    return args


ARGS = parse_args()

## fix the random seeds for reproducibility
# dr = 2

for dr in range(1,2):
    randomseed = ARGS.seed * dr
    random.seed(randomseed)
    torch.manual_seed(randomseed)
    np.random.seed(randomseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(randomseed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def _basin_norm(
        x: np.array, basin_area: np.array, to_norm: bool
    ) -> np.array:
        """

        Because units of streamflow and area are ft^3/s and km^2 respectively,
        and we need (mm/day), we transform the equation as the code shows.

        """
        nd = len(x.shape)

        if nd == 3 and x.shape[2] == 1:
            x = x[:, :, 0]  # unsqueeze the original 3 dimension matrix
        temparea = np.tile(basin_area, (1, x.shape[1]))

        if to_norm is True:

            flow = (x * 0.0283168 * 3600 * 24) / (temparea * (10 ** 6)) * 10 ** 3

        else:

            flow = (
                x
                * ((temparea * (10**6)) * (10 ** (-3)))
                / (0.0283168 * 3600 * 24)
            )

        if nd == 3:
            flow = np.expand_dims(flow, axis=2)
        return flow


    def trans_norm(x, var_lst, stat_dict, *, to_norm):
        """
        normalization, including denormalization code

        Parameters
        ----------
        x
            2d or 3d data
            2d：1st-sites，2nd-var type
            3d：1st-sites，2nd-time, 3rd-var type
        var_lst
            variables
        stat_dict
            a dict with statistics info
        to_norm
            if True, normalization; else denormalization

        Returns
        -------
        np.array
            normalized/denormalized data
        """
        if type(var_lst) is str:
            var_lst = [var_lst]
        out = np.zeros(x.shape)
        for k in range(len(var_lst)):
            var = var_lst[k]
            stat = stat_dict[var]
            if to_norm is True:
                if len(x.shape) == 3:
                    out[:, :, k] = (x[:, :, k] - stat[2]) / stat[3]
                elif len(x.shape) == 2:
                    out[:, k] = (x[:, k] - stat[2]) / stat[3]
            else:
                if len(x.shape) == 3:
                    out[:, :, k] = x[:, :, k] * stat[3] + stat[2]
                elif len(x.shape) == 2:
                    out[:, k] = x[:, k] * stat[3] + stat[2]
        return out

    def _trans_norm(
        x: np.array, var_lst: list, stat_dict: dict, log_norm_cols: list, to_norm: bool
    ) -> np.array:
        """
        Normalization or inverse normalization

        There are two normalization formulas:

        .. math:: normalized_x = (x - mean) / std

        and

         .. math:: normalized_x = [log_{10}(\sqrt{x} + 0.1) - mean] / std

         The later is only for vars in log_norm_cols; mean is mean value; std means standard deviation

        Parameters
        ----------
        x
            data to be normalized or denormalized
        var_lst
            the type of variables
        stat_dict
            statistics of all variables
        log_norm_cols
            which cols use the second norm method
        to_norm
            if true, normalize; else denormalize

        Returns
        -------
        np.array
            normalized or denormalized data
        """
        if type(var_lst) is str:
            var_lst = [var_lst]
        out = np.full(x.shape, np.nan)
        for k in range(len(var_lst)):
            var = var_lst[k]
            stat = stat_dict[var]
            if to_norm is True:
                if len(x.shape) == 3:
                    if var in log_norm_cols:
                        x[:, :, k] = np.log10(np.sqrt(x[:, :, k]) + 0.1)
                    out[:, :, k] = (x[:, :, k] - stat[2]) / stat[3]
                elif len(x.shape) == 2:
                    if var in log_norm_cols:
                        x[:, k] = np.log10(np.sqrt(x[:, k]) + 0.1)
                    out[:, k] = (x[:, k] - stat[2]) / stat[3]
            else:
                if len(x.shape) == 3:
                    out[:, :, k] = x[:, :, k] * stat[3] + stat[2]
                    if var in log_norm_cols:
                        out[:, :, k] = (np.power(10, out[:, :, k]) - 0.1) ** 2
                elif len(x.shape) == 2:
                    out[:, k] = x[:, k] * stat[3] + stat[2]
                    if var in log_norm_cols:
                        out[:, k] = (np.power(10, out[:, k]) - 0.1) ** 2
        return out


    def hargreaves(tmin, tmax, tmean, lat, trange):
        # calculate the day of year
        dfdate = pd.date_range(start=str(trange[0]), end=str(trange[1]), freq='D', closed='left') # end not included
        tempday = np.array(dfdate.dayofyear)
        day_of_year = np.tile(tempday.reshape(-1, 1), [1, tmin.shape[-1]])
        # Loop to reduce memory usage
        pet = np.zeros(tmin.shape, dtype=np.float32) * np.NaN
        for ii in np.arange(len(pet[:, 0])):
            trange = tmax[ii, :] - tmin[ii, :]
            trange[trange < 0] = 0

            latitude = np.deg2rad(lat[ii, :])

            SOLAR_CONSTANT = 0.0820

            sol_dec = 0.409 * np.sin(((2.0 * np.pi / 365.0) * day_of_year[ii, :] - 1.39))

            sha = np.arccos(np.clip(-np.tan(latitude) * np.tan(sol_dec), -1, 1))

            ird = 1 + (0.033 * np.cos((2.0 * np.pi / 365.0) * day_of_year[ii, :]))

            tmp1 = (24.0 * 60.0) / np.pi
            tmp2 = sha * np.sin(latitude) * np.sin(sol_dec)
            tmp3 = np.cos(latitude) * np.cos(sol_dec) * np.sin(sha)
            et_rad = tmp1 * SOLAR_CONSTANT * ird * (tmp2 + tmp3)

            pet[ii, :] = 0.0023 * (tmean[ii, :] + 17.8) * trange ** 0.5 * 0.408 * et_rad

        pet[pet < 0] = 0

        return pet

    def cal_4_stat_inds(b):
        """
        Calculate four statistics indices: percentile 10 and 90, mean value, standard deviation

        Parameters
        ----------
        b
            input data

        Returns
        -------
        list
            [p10, p90, mean, std]
        """
        p10 = np.percentile(b, 10).astype(float)
        p90 = np.percentile(b, 90).astype(float)
        mean = np.mean(b).astype(float)
        std = np.std(b).astype(float)
        if std < 0.001:
            std = 1
        return [p10, p90, mean, std]


    def cal_stat_gamma(x):
        """
        Try to transform a time series data to normal distribution

        Now only for daily streamflow, precipitation and evapotranspiration;
        When nan values exist, just ignore them.

        Parameters
        ----------
        x
            time series data

        Returns
        -------
        list
            [p10, p90, mean, std]
        """
        a = x.flatten()
        b = a[~np.isnan(a)]  # kick out Nan
        b = np.log10(
            np.sqrt(b) + 0.1
        )  # do some tranformation to change gamma characteristics
        return cal_4_stat_inds(b)

    def cal_stat(x: np.array) -> list:
        """
        Get statistic values of x (Exclude the NaN values)

        Parameters
        ----------
        x: the array

        Returns
        -------
        list
            [10% quantile, 90% quantile, mean, std]
        """
        a = x.flatten()
        b = a[~np.isnan(a)]
        if b.size == 0:
            # if b is [], then give it a 0 value
            b = np.array([0])
        return cal_4_stat_inds(b)

    def fill_Nan(array_3d):
        # Define the x-axis for interpolation
        x = np.arange(array_3d.shape[1])

        # Iterate over the first and third dimensions to interpolate the second dimension
        for i in range(array_3d.shape[0]):
            for j in range(array_3d.shape[2]):
                # Select the 1D slice for interpolation
                slice_1d = array_3d[i, :, j]

                # Find indices of NaNs and non-NaNs
                nans = np.isnan(slice_1d)
                non_nans = ~nans

                # Only interpolate if there are NaNs and at least two non-NaN values for reference
                if np.any(nans) and np.sum(non_nans) > 1:
                    # Perform linear interpolation using numpy.interp
                    array_3d[i, :, j] = np.interp(x, x[non_nans], slice_1d[non_nans], left=None, right=None)
        return array_3d
    ## GPU setting
    # which GPU to use when having multiple
    traingpuid = ARGS.gpu
    testfoldInd = 1
    test_other=0
    test_temporal = 1
    ALLepoch = 100
    testepoch = ARGS.epoch
    HIDDENSIZE = 256
    if torch.cuda.is_available():
        torch.cuda.set_device(traingpuid)

    dataset_select=np.load(Path(ARGS.dataset_npz).expanduser())
    # [nx,ngage,ntime]
    # [nx,ngage,1]
    print(dataset_select.files)

    ## Time span of Daymet
    AllTime = pd.date_range('2015-04-01', f'2020-03-31', freq='d')
    # train_time_bgn = pd.datetime('2015-04-01')
    time_train_bgn = pd.to_datetime("2015-04-01")
    time_train_end = pd.to_datetime('2018-03-31')
    time_warmup_bgn = pd.to_datetime('2015-04-01')
    time_test_bgn = pd.to_datetime('2018-04-01')
    time_test_end = pd.to_datetime('2020-03-31')

    index_warmup = AllTime.get_loc(time_warmup_bgn)
    index_start = AllTime.get_loc(time_test_bgn)
    index_end = AllTime.get_loc(time_test_end)+1

    warmup_span = pd.date_range(time_warmup_bgn,time_test_bgn, freq='d', closed=None)
    warmup_lens = len(warmup_span)

    ## Slice Daymet data for the time span where we have streamflow data
    forcingAll = dataset_select['data_ts'][:-2].transpose(1,2,0)[:,index_warmup:index_end,:]
    # ## IDs of gages where we have streamflow data
    # shapeID_str_lst= train_data_dict['sites_id']

    ## Fill the nans in Daymet with interpolation (This is for missing data in leap years)
    forcingAll = fill_Nan(forcingAll)

    forcingAllLst  = dataset_select['name_ts'][:-2]
    ##Streamflow data
    streamflow_train = dataset_select['data_ts'][-2][:,:,None][:,index_warmup:index_end,:]
    streamflow_train_list = [dataset_select['name_ts'][-2]]


    # warmup_span = pd.date_range(train_time_bgn,train_time_end, freq='d', closed=None)
    # warmup_for_testing = len(warmup_span)
    # streamflow = streamflow_train[:,:warmup_for_testing,:]
    streamflow = streamflow_train

    ###############################################################################################################
    ## Attributes collected by Jiangtao
    attributeALL_df = dataset_select['data_st'].transpose(1,0,2).squeeze()

    attributeAllLst = dataset_select['name_st']

    forcingAll = forcingAll[:,:,:]
    streamflow = streamflow[:,:,:]

    attributeLst = attributeAllLst

    forcing_HBV_List = dataset_select['name_ts'][:-2]

    ## Attributes collected by Jiangtao
    attribute = attributeALL_df
    attributeLst = attributeAllLst

    pet_raw = pd.read_csv(Path(ARGS.pet_csv).expanduser()).values[:,1:]
    pet_raw[pet_raw<0]=0
    pet = pet_raw[dataset_select['selected_id'],index_warmup:index_end]

    xTrain = forcingAll
    ##Build the input for HBV for parameterization

    out = str(Path(ARGS.model_dir).expanduser())
    result_out = Path(ARGS.output_dir).expanduser()
    result_out.mkdir(parents=True, exist_ok=True)
    with open(Path(out) / 'dapengscaler_stat.json') as f:
        stat_dict = json.load(f)

    log_norm_cols=['prcp_log', 'pr', 'total_precipitation', 'pre', 'potential_evaporation', 'LE',
                       'PLE', 'GPP', 'Ec', 'Es', 'Ei', 'ET_water', 'ET_sum', 'susm', 'smp', 'ssma', 'susma',
                       'usgsFlow', 'streamflow', 'qobs']

    ##Build the input for LSTM for parameterization
    forcing_LSTM_norm = _trans_norm(
        xTrain.copy(), forcingAllLst, stat_dict, log_norm_cols=log_norm_cols, to_norm=True
    )
    xTrain[xTrain!=xTrain] = 0
    forcing_LSTM_norm[forcing_LSTM_norm!=forcing_LSTM_norm] = 0

    streamflow_trans = streamflow
    forcing_HBV = np.concatenate((xTrain[:,:,[0,5]],pet[:,:,None]),axis=2)
    forcing_HBV[forcing_HBV!=forcing_HBV] = 0

    attribute_norm = trans_norm(attribute, list(attributeLst), stat_dict, to_norm=True)
    attribute_norm[attribute_norm!=attribute_norm] = 0

    attribute_norm = np.expand_dims(attribute_norm, axis=1)
    attribute_norm = np.repeat(attribute_norm, forcing_LSTM_norm.shape[1], axis=1)

    # streamflow_trans = _trans_norm(
    #     streamflow_train.copy(), streamflow_train_list, stat_dict, log_norm_cols=log_norm_cols, to_norm=True
    # )
    # forcTuple = [forcing_HBV,forcing_LSTM_norm]

    # hyper-parameters
    Nmul=16  # Multi-component model. How many parallel HBV components to use. 1 means the original HBV.
    Ninv=forcing_LSTM_norm.shape[-1]+attribute_norm.shape[-1]
    Nfea=13+1+2  # should be 13 when setting ETMod=True. 12 when ETMod=False
    dydrop=0.0  # dropout possibility for those dynamic parameters: 0.0 always dynamic; 1.0 always static
    routing=False  # Routing is intentionally disabled for the soil-moisture study.
    comprout=False  # True is doing routing for each component
    compwts=False  # True is using weighted average for components; False is the simple mean
    staind=-1  # which time step to use from the learned para time series for those static parameters

    tdRep=[1,3,13]  # When using dynamic parameters, this list defines which parameters to set as dynamic
    ETMod=True

    testmodel=rnn.MultiInv_HBVTDModel_u30(ninv=Ninv,nfea=Nfea,nmul=Nmul,hiddeninv=HIDDENSIZE,
                                      routOpt=routing,comprout=comprout,compwts=compwts,staind=staind,tdlst=tdRep,
                                      dydrop=dydrop,ETMod=ETMod,
                                      model_name='HBV_Smap',
                                      model_name_sub='HBVMulTDET_1_1p_no_g_SMAP_ab')

    model_path = out
    print("Load model from ", model_path)
    # Specify the GPU device to use
    device = torch.device(f'cuda:{traingpuid }'  if torch.cuda.is_available() else 'cpu')
    # testmodel = loadModel(model_path, epoch=testepoch,map_location=device)
    testmodel.load_state_dict(loadStateDict(model_path,epoch=testepoch,map_location=device))

    zTest = np.concatenate([forcing_LSTM_norm, attribute_norm], 2)  # Add attributes to historical forcings as the inversion part
    xTest = forcing_HBV
    testTuple = (xTest[:], zTest[:])
    testbatch =150 #len(indexes)

    filePathLst = [str(result_out / f"out{k}") for k in range(5)]


    if test_temporal == 1:
        testmodel.inittime = 0
    else:
        testmodel.inittime=5478
    train.testModel(
        testmodel, testTuple, c=None, batchSize=testbatch, filePathLst=filePathLst)


    dataPred = pd.read_csv(result_out / "out0", dtype=np.float32, header=None).values
    dataPred = np.expand_dims(dataPred, axis=-1)

    if test_temporal ==1 or test_temporal == 0:
        evaDict = [stat.statError(dataPred[:,warmup_lens:,0], streamflow_trans[:,warmup_lens:,0])]
        # evaDict=[stat.statError(dataPred[:,:warmup_lens,0],streamflow_trans[:,:warmup_lens,0])]

    evaDictLst = evaDict
    # keyLst = ['NSE', 'KGE','FLV','FHV', 'lowRMSE', 'highRMSE','rdMax','absFLV','absFHV']
    keyLst = ['NSE', 'KGE','FLV','FHV', 'lowRMSE', 'highRMSE']
    dataBox = list()
    for iS in range(len(keyLst)):
        statStr = keyLst[iS]
        temp = list()
        for k in range(len(evaDictLst)):
            data = evaDictLst[k][statStr]
            #data = data[~np.isnan(data)]
            temp.append(data)
        dataBox.append(temp)


    # print("LSTM model'NSE', 'KGE','FLV','FHV', 'lowRMSE', 'highRMSE' ,'rdMax','absFLV','absFHV'",
    #       np.nanmedian(dataBox[0][0]),
    #       np.nanmedian(dataBox[1][0]), np.nanmedian(dataBox[2][0]), np.nanmedian(dataBox[3][0]),
    #       np.nanmedian(dataBox[4][0]), np.nanmedian(dataBox[5][0]), np.nanmedian(dataBox[6][0]), np.nanmedian(dataBox[7][0]), np.nanmedian(dataBox[8][0]))
    print("LSTM model'NSE', 'KGE','FLV','FHV', 'lowRMSE', 'highRMSE'",
          np.nanmedian(dataBox[0][0]),
          np.nanmedian(dataBox[1][0]), np.nanmedian(dataBox[2][0]), np.nanmedian(dataBox[3][0]),
          np.nanmedian(dataBox[4][0]), np.nanmedian(dataBox[5][0]))
