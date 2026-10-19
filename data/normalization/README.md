# Normalization files

Training computes normalization statistics from the supplied inputs and saves
`dapengscaler_stat.json` in the experiment directory. Testing uses the matching
file from that experiment together with its trained checkpoint.
