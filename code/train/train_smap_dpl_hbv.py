import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))

from hydroDL.model import crit, train
from hydroDL.model import rnn as rnn
import argparse
import os
import numpy as np
import torch
import random
import pandas as pd
import json
import yaml

from sklearn.model_selection import KFold


def parse_args():
    default_config = Path(__file__).resolve().parents[1] / 'configs' / 'example_config.yaml'
    parser = argparse.ArgumentParser(
        description='Portable training entry point for the SMAP dPL-HBV reference package.'
    )
    parser.add_argument('--config', default=str(default_config), help='YAML configuration file.')
    parser.add_argument('--dataset-npz', help='Combined NPZ with dynamic data, static data, names, and selected IDs.')
    parser.add_argument('--pet-csv', help='PEVAP CSV aligned to the source grid and experiment dates.')
    parser.add_argument('--output-dir', help='Directory under which the experiment folder is created.')
    parser.add_argument('--gpu', type=int, default=0, help='CUDA device index; ignored without CUDA.')
    parser.add_argument('--seed', type=int, default=111111, help='Random seed.')
    known, _ = parser.parse_known_args()
    config_path = Path(known.config)
    if config_path.exists():
        with config_path.open(encoding='utf-8') as stream:
            config = yaml.safe_load(stream) or {}
        parser.set_defaults(**config.get('train', {}))
    args = parser.parse_args()
    missing = [name for name in ('dataset_npz', 'pet_csv', 'output_dir') if not getattr(args, name)]
    if missing:
        parser.error('missing configuration values: ' + ', '.join(missing))
    return args


ARGS = parse_args()

## fix the random seeds for reproducibility
randomseed = ARGS.seed
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
if torch.cuda.is_available():
    torch.cuda.set_device(traingpuid)

## streamflow data I extracted from USGS
dataset_select = np.load(Path(ARGS.dataset_npz).expanduser())
# [nx,ngage,ntime]
# [nx,ngage,1]
print(dataset_select.files)

## Time span of Daymet
AllTime = pd.date_range('2015-04-01', f'2020-03-31', freq='d')
# train_time_bgn = pd.datetime('2015-04-01')
train_time_bgn = pd.to_datetime("2015-04-01")
train_time_end = pd.to_datetime('2018-03-31')
index_start = AllTime.get_loc(train_time_bgn)
index_end = AllTime.get_loc(train_time_end)

## Slice Daymet data for the time span where we have streamflow data
forcingAll = dataset_select['data_ts'][:-2].transpose(1,2,0)[:,index_start:index_end,:]
# ## IDs of gages where we have streamflow data
# shapeID_str_lst= train_data_dict['sites_id']

## Fill the nans in Daymet with interpolation (This is for missing data in leap years)
forcingAll = fill_Nan(forcingAll)

forcingAllLst  = dataset_select['name_ts'][:-2]
##Streamflow data
streamflow_train = dataset_select['data_ts'][-2][:,:,None][:,index_start:index_end,:]
streamflow_train_list = [dataset_select['name_ts'][-2]]


warmup_span = pd.date_range(train_time_bgn,train_time_end, freq='d', closed=None)
warmup_for_testing = len(warmup_span)
streamflow = streamflow_train[:,:warmup_for_testing,:]

###############################################################################################################
## Attributes collected by Jiangtao
attributeALL_df = dataset_select['data_st'].transpose(1,0,2).squeeze()

attributeAllLst = dataset_select['name_st']

forcingAll = forcingAll[:,:,:]
streamflow = streamflow[:,:,:]


log_norm_cols=['prcp_log', 'pr', 'total_precipitation', 'pre', 'potential_evaporation', 'LE',
                   'PLE', 'GPP', 'Ec', 'Es', 'Ei', 'ET_water', 'ET_sum', 'susm', 'smp', 'ssma', 'susma',
                   'usgsFlow', 'streamflow', 'qobs']

attributeLst = attributeAllLst

forcing_HBV_List = dataset_select['name_ts'][:-2]

##Caculate the statistics for forcing
stat_dict={}
for fid, forcing_item in enumerate(forcingAllLst):
    if forcing_item in log_norm_cols:
        stat_dict[forcing_item] = cal_stat_gamma(forcingAll[:,:,fid])
    else:
        stat_dict[forcing_item] = cal_stat(forcingAll[:,:,fid])

for fid, forcing_item in enumerate(streamflow_train_list):
    if forcing_item in log_norm_cols:
        stat_dict[forcing_item] = cal_stat_gamma(streamflow_train[:,:,fid])
    else:
        stat_dict[forcing_item] = cal_stat(streamflow_train[:,:,fid])

## Slice the attributes we want from all attributes

attribute = attributeALL_df

attributeLst = attributeAllLst


# ####################################################################################################
# ## load PUB list
# splitPath = '../data/CONUS_3200/py02_PUBsplitLst_CONUS.txt'
# with open(splitPath, 'r') as fp:
#     testIDLst = json.load(fp)
# TestLS = testIDLst[testfoldInd - 1]
# TrainLS = list(set(idLst_new.tolist()) - set(TestLS))
#
# [C, ind1, PUB_SubInd] = np.intersect1d(TrainLS, idLst_new, return_indices=True)
#
# #####################################################################################################





##Caculate the statistics for attributes
for aid, attribute_item in enumerate (attributeLst):
    stat_dict[attribute_item] = cal_stat(attribute[:,aid])

##Use tmean to caculate PET
# Tex = [19801001, 19951001]
# Tex = train_data_dict["t_final_range"]

# tmin = np.swapaxes(forcingAll[:,:,np.where(np.array(forcingAllLst) == "tmin")[0][0]], 0,1)
# tmax = np.swapaxes(forcingAll[:,:,np.where(np.array(forcingAllLst) == "tmax")[0][0]], 0,1)
#
# tmean = (tmin+tmax)/2
#
# latarray = np.tile(lat, [tmin.shape[0], 1])
# pet = hargreaves(tmin, tmax, tmean, lat=latarray, trange=Tex)

pet_raw = pd.read_csv(Path(ARGS.pet_csv).expanduser()).values[:,1:]
pet_raw[pet_raw<0]=0
pet = pet_raw[dataset_select['selected_id'],index_start:index_end]



##Reorganize the forcing ['prcp','tmean','pet']
# xTrain = np.zeros([forcingAll.shape[0],forcingAll.shape[1],3])
xTrain = forcingAll
# xTrain[:,:,1] = np.swapaxes(tmean, 0,1)
# xTrain[:,:,2] = np.swapaxes(pet, 0,1)


# streamflow_trans = _basin_norm(
#                         streamflow[:, :, 0 :  1].copy(), basin_area, to_norm=True
#                     )  ## from ft^3/s to mm/day


## calculate the statistics for new forcing terms
# stat_dict['tmean'] = cal_stat(xTrain[:,:,1])
stat_dict['pet'] = cal_stat(pet)

##Build the input for LSTM for parameterization
forcing_LSTM_norm = _trans_norm(
    xTrain.copy(), forcingAllLst, stat_dict, log_norm_cols=log_norm_cols, to_norm=True
)

# streamflow_trans = _trans_norm(
#     streamflow_train.copy(), streamflow_train_list, stat_dict,to_norm=True
# )
streamflow_trans = streamflow

forcing_HBV = np.concatenate((xTrain[:,:,[0,5]],pet[:,:,None]),axis=2)
xTrain[xTrain!=xTrain] = 0
forcing_LSTM_norm[forcing_LSTM_norm!=forcing_LSTM_norm] = 0
forcing_HBV[forcing_HBV!=forcing_HBV] = 0

attribute_norm = trans_norm(attribute, list(attributeLst), stat_dict, to_norm=True)
attribute_norm[attribute_norm!=attribute_norm] = 0


Ninv = forcing_LSTM_norm.shape[-1]+attribute_norm.shape[-1]
EPOCH = 100 # total epoches to train the mode
# EPOCH = 2 # total epoches to train the mode
BATCH_SIZE = 100
RHO = 365
saveEPOCH = 10
# saveEPOCH = 1
alpha = 0.25
HIDDENSIZE = 256
BUFFTIME = 365 # for each training sample, to use BUFFTIME days to warm up the states.
routing = False # Routing is intentionally disabled for the soil-moisture study.
Nmul = 16 # Multi-component model. How many parallel HBV components to use. 1 means the original HBV.
comprout = False # True is doing routing for each component
compwts = False # True is using weighted average for components; False is the simple mean
pcorr = None # or a list to give the range of precip correc

tdRep = [1, 3, 13]  # When using dynamic parameters, this list defines which parameters to set as dynamic
tdRepS = [str(ix) for ix in tdRep]
# ETMod: if True, use the added shape parameter (index 13) for ET. Default as False.
# Must set below ETMod as True and Nfea=13 when including 13 index in above tdRep list for dynamic parameters
# If 13 not in tdRep list, set below ETMod=False and Nfea=12 to use the original HBV without ET shape para
ETMod = True
Nfea = 13+1+2  # should be 13 when setting ETMod=True. 12 when ETMod=False
dydrop = 0.0  # dropout possibility for those dynamic parameters: 0.0 always dynamic; 1.0 always static
staind = -1  # which time step to use from the learned para time series for those static parameters


model=rnn.MultiInv_HBVTDModel_u30(ninv=Ninv,nfea=Nfea,nmul=Nmul,hiddeninv=HIDDENSIZE,inittime=BUFFTIME,
                                  routOpt=routing,comprout=comprout,compwts=compwts,staind=staind,tdlst=tdRep,
                                  dydrop=dydrop,ETMod=ETMod,
                                  model_name='HBV_Smap',
                                  model_name_sub='HBVMulTDET_1_1p_no_g_SMAP_ab')


# lossFun = crit.RmseLossComb(alpha=alpha)
lossFun = crit.NSELossBatch_song(np.nanstd(streamflow_trans, axis=1))



rootOut = str(Path(ARGS.output_dir).expanduser())
if os.path.exists(rootOut) is False:
    os.makedirs(rootOut)
    # os.makedirs(rootOut) # can make more directions
out = os.path.join(rootOut, f"exp_EPOCH{EPOCH}_BS{BATCH_SIZE}_RHO{RHO}_HS{HIDDENSIZE}_trainBuff{BUFFTIME}") # output folder to save results
if os.path.exists(out) is False:
    os.mkdir(out)

with open(out+'/dapengscaler_stat.json','w') as f:
    json.dump(stat_dict, f)

# from hydroDL.master import loadStateDict
# model_path = out
# testepoch=100
# model.load_state_dict(loadStateDict(model_path,epoch=testepoch))


##K fold for spatial tests
# fold_number = 1
# # Define the number of folds (K)
# k = 6
#
# # Initialize the KFold cross-validator
# kf = KFold(n_splits=k, shuffle=True, random_state=randomseed)
#
# # Iterate through the folds
# for fold_number_idx, (train_index, test_index) in enumerate(kf.split(streamflow_trans)):
#     if fold_number_idx == fold_number:
#         xTrain_fold = xTrain[train_index,:,:]
#         forcing_LSTM_norm_fold = forcing_LSTM_norm[train_index,:,:]
#         streamflow_trans_fold = streamflow_trans[train_index,:,:]
#         attribute_norm_fold = attribute_norm[train_index,:]
# forcTuple_fold = [xTrain_fold,forcing_LSTM_norm_fold]
#
# trainedModel = train.trainModel(
#     model,
#     forcTuple_fold,
#     streamflow_trans_fold,
#     attribute_norm_fold,
#     lossFun,
#     nEpoch=EPOCH,
#     miniBatch=[BATCH_SIZE, RHO],
#     saveEpoch=saveEPOCH,
#     saveFolder=out,
#     bufftime=BUFFTIME)

# For temporal test
# forcTuple = [xTrain[PUB_SubInd],forcing_LSTM_norm[PUB_SubInd]]
forcTuple = [forcing_HBV,forcing_LSTM_norm]
# forcingAll = forcingAll[PUB_SubInd,:,:]
# streamflow = streamflow[PUB_SubInd,:,:]
# attribute = attribute[PUB_SubInd,:]
# basin_area = basin_area[PUB_SubInd,:]
# lat = lat[PUB_SubInd]

# Check whether all array elements are NaN
arr = streamflow_trans
if np.all(np.isnan(arr)):
    print("All elements in the array are NaN")
else:
    print("The array contains non-NaN elements")
arr = forcing_LSTM_norm
# Check whether the array contains NaN
if np.any(np.isnan(arr)):
    print("The array contains NaN")
else:
    print("The array contains no NaN")

# Find the positions of NaN values
nan_positions = np.where(np.isnan(arr))

trainedModel = train.trainModel(
    model,
    forcTuple,
    streamflow_trans,
    attribute_norm,
    lossFun,
    nEpoch=EPOCH,
    miniBatch=[BATCH_SIZE, RHO],
    saveEpoch=saveEPOCH,
    saveFolder=out,
    bufftime=BUFFTIME)
