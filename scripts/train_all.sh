#!/bin/sh

for eos in *.h5; do
  stem=$(basename "$eos" .h5)
  pyghl train "$eos" \
    --append_eos no \
    --register_installed_model yes \
    --overwrite_installed_model \
    --hdf5_output "/tmp/${stem}_nn.h5" \
    --bundle_output "/tmp/${stem}_nn.pt" \
    --header_output "/tmp/${stem}_nn.h" \
    --log_path "/tmp/${stem}_training.log" \
    --checkpoint_dir "/tmp/${stem}_checkpoints"
done
