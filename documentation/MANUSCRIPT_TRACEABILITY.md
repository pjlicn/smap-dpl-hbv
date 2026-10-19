# Manuscript traceability

Line numbers below refer to files in this archive.

| Concept | File and lines |
|---|---|
| LSTM_dPL and direct-HBV coupling | `code/model/hydroDL/model/rnn.py:8101-8183` |
| Parameter sigmoid and 16-component reshape | `code/model/hydroDL/model/rnn.py:8160-8181` |
| SMAP-tailored HBV class | `code/model/hydroDL/model/HydroModels/HBV_Smap.py:516-762` |
| HBV parameter bounds | `code/model/hydroDL/model/HydroModels/HBV_Smap.py:585-592` |
| Dynamic/static parameter assembly | `code/model/hydroDL/model/HydroModels/HBV_Smap.py:596-606` |
| HBV state and flux equations | `code/model/hydroDL/model/HydroModels/HBV_Smap.py:625-700` |
| Soil moisture divided by field capacity | `code/model/hydroDL/model/HydroModels/HBV_Smap.py:656-680` |
| Linear correction and a_LC/b_LC roles | `code/model/hydroDL/model/HydroModels/HBV_Smap.py:631,707` |
| Corrected soil-moisture component average | `code/model/hydroDL/model/HydroModels/HBV_Smap.py:709-714` |
| Direct model outputs | `code/model/hydroDL/model/HydroModels/HBV_Smap.py:756-762` |
| Soil-moisture study routing configuration | `code/train/train_smap_dpl_hbv.py:469-471`; `code/test/test_smap_dpl_hbv.py:431-433` |
| LSTM_SURR architecture | `code/model/hydroDL/model/HydroModels/HBV_Smap.py:11-31` |
| Surrogate dPL wrapper | `code/model/hydroDL/model/pjli_model.py:3505-3589` |
| Surrogate HBV adapter and correction | `code/model/hydroDL/model/HydroModels/HBV_Smap.py:862-958` |
| Static attribute order and normalization functions | `code/train/train_smap_dpl_hbv.py:316-404,455-456` |
| APCP/TMP/PET HBV forcing | `code/train/train_smap_dpl_hbv.py:363-420,450-453` |
| Training dates and slices | `code/train/train_smap_dpl_hbv.py:321-330` |
| Sequence/warm-up configuration | `code/train/train_smap_dpl_hbv.py:459-490,546-579` |
| Test dates, warm-up, and output evaluation | `code/test/test_smap_dpl_hbv.py:335-368,460-472` |
| NaN-masked SMAP loss | `code/model/hydroDL/model/crit.py:291-317` |
| Training loop and checkpoint writing | `code/model/hydroDL/model/train.py:10-212` |
| Test batching and output writing | `code/model/hydroDL/model/train.py:1192-1360` |
| Evaluation metrics | `code/model/hydroDL/post/stat.py:88-180` and continuation |
| Static-data lossless extraction | `code/utils/extract_static_attributes.py` |

The table maps the model components and entry points in this package to the
methods described in the paper.
