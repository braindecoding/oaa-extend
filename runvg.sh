#!/bin/bash

# Define parameter arrays
intermediate_dims=(128 256 512)
batch_sizes=(10 20 30 40 50)
# max_iters=(500 1000 1500)
max_iters=(2000 2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 8000 8500 9000 9500 10000 10500 11000 11500 12000 12500 13000)
# latent=(3 6 12 18)
latent=(9 15)

# Loop over latent
for z in "${latent[@]}"; do
    # Loop over intermediate dimensions
    for i in "${intermediate_dims[@]}"; do
        # Loop over batch sizes
        for j in "${batch_sizes[@]}"; do
            # Loop over max iterations
            for k in "${max_iters[@]}"; do
                # Run scripts
                python oaavangerven.py "$z" "$i" "$j" "$k"
                python fidvg.py "$z" "$i" "$j" "$k"
            done
        done
    done
done
