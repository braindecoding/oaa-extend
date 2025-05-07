@echo off

setlocal enabledelayedexpansion

rem Define parameters
set "intermediate_dims=(128 256 512)"
set "batch_sizes=(10 20 30 40 50)"
set "max_iters=(500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 8000 8500 9000 9500 10000 10500 11000 11500 12000 12500 13000)"
rem set "latent=(3 6 12 18)"
set "latent=(9 15)"

rem Loop over latent
for %%z in %latent% do (
    rem Loop over intermediate dimensions
    for %%i in %intermediate_dims% do (
        rem Loop over batch sizes
        for %%j in %batch_sizes% do (
            rem Loop over max iterations
            for %%k in %max_iters% do (
                rem Run scripts
                python oaamiyawaki.py %%z %%i %%j %%k
				python fidmy.py %%z %%i %%j %%k
            )
        )
    )
)
