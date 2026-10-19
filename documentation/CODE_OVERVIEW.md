# Code overview

## Architecture

The primary model is `MultiInv_HBVTDModel_u30` in
`code/model/hydroDL/model/rnn.py`. A one-layer custom CUDNN LSTM maps 14 inputs
(six atmospheric variables and eight repeated static attributes) to:

- 16 parameters for each of 16 parallel HBV components;
- two routing parameters.

The routing slots and routing-capable model interfaces are retained for
generic compatibility. The soil-moisture study configuration sets routing to
false, so runoff routing is not activated in the reported configuration.

The first 256 outputs are reshaped to
`[time, location, parameter, component]` after a sigmoid. The selected
`HBVMulTDET_1_1p_no_g_SMAP_ab` class transforms normalized parameters to their
configured ranges and advances the HBV states.

Parameter order and bounds:

| Index | Parameter | Bounds | Primary setting |
|---:|---|---:|---|
| 1 | BETA | [1, 6] | dynamic |
| 2 | FC | [50, 1000] | static |
| 3 | K0 | [0.05, 0.9] | dynamic |
| 4 | K1 | [0.01, 0.5] | static |
| 5 | K2 | [0.001, 0.2] | static |
| 6 | LP | [0.2, 1] | static |
| 7 | PERC | [0, 10] | static |
| 8 | UZL | [0, 100] | static |
| 9 | TT | [-2.5, 2.5] | static |
| 10 | CFMAX | [0.5, 10] | static |
| 11 | CFR | [0, 0.1] | static |
| 12 | CWH | [0, 0.2] | static |
| 13 | BETAET | [0.3, 5] | dynamic |
| 14 | C | [0, 1] | static |
| 15 | a_LC | [0, 5] | static |
| 16 | b_LC | [-1, 1] | static |

“Static” here means that the model uses the parameter value from `staind=-1`
across the sequence. It does not mean the parameter is a fixed global
constant.

## SMAP output

At each time step and component, HBV calculates snow, soil, upper-zone, and
lower-zone states. The SMAP-tailored output is:

```text
SM_HBV = clamp((SM / FC - b_LC) * a_LC, 0, 1)
```

The 16 component results are averaged. The returned tensor also contains Q0,
Q1, Q2, actual ET, and SWE.

## LSTM surrogate

`CudnnLstmModel_SMAP` is a one-layer surrogate with a rectified input
projection, custom CUDNN LSTM, output projection, and `[0,1]` output clamp.
`MultiInv_HBVTDModel_u30_surrogate` connects LSTM_dPL to a surrogate HBV
adapter. These classes provide the model components for surrogate-based
parameter learning and reuse of the model architecture.

## Training and testing utilities

`trainModel` uses Adadelta, random location/time minibatches, and a preceding
warm-up segment. `NSELossBatch_song` masks non-finite targets represented by
NaN and applies basin-standard-deviation weighting. `testModel` evaluates
location batches and writes one CSV per requested output variable.

## Portability notes

| Component | Adaptation | Purpose |
|---|---|---|
| Training and testing entry points | Package-relative imports, YAML/CLI paths, configurable GPU and seed | Configure local inputs and experiment directories |
| `rnn.py`, `pjli_model.py` | Device-aware constructor initialization | Construct the model for inspection on CPU; custom LSTM execution uses CUDA/CUDNN |
| `utils/interp.py` | String equality comparisons | Compatibility and correctness of string comparisons |
| Package `__init__.py` files | Imports limited to the packaged components | Use the package without unrelated cluster defaults |

Model equations, parameter ranges, sequence settings, and warm-up behavior
are described above and in `REPRODUCIBILITY_NOTES.md`. Package verification
covers construction and input integrity; numerical prediction equivalence is
outside that verification scope.
