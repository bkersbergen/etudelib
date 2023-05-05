

### How to run?
`make` displays all targets

`make configure` setup a virtual environment and install regular pytorch 2.0

`make run` executes the Rust application to load a model and print a prediction output to console


### Troubleshooting tch-rs (Mac M1)
thread 'main' panicked at 'Pre-built version of libtorch for apple silicon are not available.
You can install torch manually following the indications from https://github.com/LaurentMazare/tch-rs/issues/629

`pip3 install torch==2.0.0`

Then update the following environment variables:
`export LIBTORCH=$(python3 -c 'import torch; from pathlib import Path; print(Path(torch.__file__).parent)')
export DYLD_LIBRARY_PATH=${LIBTORCH}/lib`

note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace


