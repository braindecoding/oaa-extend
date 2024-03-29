@echo off

setlocal enabledelayedexpansion

rem Define parameters
set "intermediate_dims=(128 256 512)"
set "batch_sizes=(10 20 30 40 50)"
set "max_iters=(500 1000 1500)"
set "latent=(3 6 12 18)"

rem Loop over latent
for %%z in %latent% do (
    rem Loop over intermediate dimensions
    for %%i in %intermediate_dims% do (
        rem Loop over batch sizes
        for %%j in %batch_sizes% do (
            rem Loop over max iterations
            for %%k in %max_iters% do (
                rem Run scripts
                python dgmmvangerven.py %%z %%i %%j %%k
            )
        )
    )
)
