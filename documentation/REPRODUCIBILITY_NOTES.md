# Model setup and reproducibility notes

## Package components and use

| Component | Use | Required inputs |
|---|---|---|
| Direct δHBV and LSTM parameter generator | Model construction and differentiable parameter learning | Prepared forcing, target, PET, and static attributes for training |
| LSTM surrogate components | Surrogate architecture and parameter-generator coupling | Surrogate weights and a configured workflow for surrogate-based experiments |
| Training entry point | Train the reference direct-model configuration | Dynamic NPZ, PET CSV, CUDA/CUDNN environment, and output paths |
| Testing entry point | Evaluate a trained model | Dynamic inputs, checkpoint, and matching scaler |
| Processed static attributes | Inspect and reuse the selected geographic attributes | NetCDF, selected IDs, and coordinates included in the package |

## Reference configuration

- Training period: 2015-04-01 through 2018-03-30.
- Test period: 2018-04-01 through 2020-03-31.
- Random seed: 111111.
- Epochs: 100.
- Batch size: 100.
- Loss-bearing sequence: 365 days.
- Warm-up: 365 days.
- LSTM hidden size: 256.
- HBV components: 16.
- Dynamic parameter indices: 1, 3, 13 (one-based).
- ET modification: enabled.
- Dynamic-parameter dropout: 0.
- Runoff routing: disabled for the soil-moisture configuration.
- Optimizer: Adadelta.

The training input period includes the warm-up data. Loss-bearing sequences
follow their preceding 365-day warm-up segment.

## Training inputs

1. A dynamic NPZ following the documented
   `data_ts/data_st/name_ts/name_st/selected_id` schema.
2. A grid-by-time PEVAP CSV aligned with the source grid and daily labels.
3. A CUDA/CUDNN-compatible PyTorch environment.
4. Input and output paths set in the configuration or command-line options.

## Testing inputs

Set the model-directory path in `run_test_example.sh` to an experiment
containing:

- `model_Ep_coeff<EPOCH>.pt`;
- the matching `dapengscaler_stat.json`.

Training creates normalization statistics from the supplied inputs. Testing
uses the statistics from the same training experiment.

## Verification

See `VERIFICATION_REPORT.md` for the package checks and their scope. The
reference configuration and model components support reuse of the core
methods; experiment-specific data and trained parameters are supplied by the
user.
