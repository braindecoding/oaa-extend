# OAA  

**High-Quality Reconstruction of fMRI Visual Stimuli Using an Optimized Autoencoder Architecture**  

FID calculation must be done separately because it utilizes multi-threading, so it cannot be combined into a single file.  

## Van Gerven Dataset  

To run tests using the Van Gerven dataset, execute:  

```sh
runvg.bat
```  

This script will run the Python scripts `dgmmvangerven.py` and `fidvg.py`.  

- **`dgmmvangerven.py`**: Performs image reconstruction with MSE error, generating an output folder for each hyperparameter, which contains subfolders with the reconstruction results.  
- **`fidvg.py`**: Calculates the FID (Fréchet Inception Distance) from the `stim` folder, which contains the original stimuli, and the `rec` folder, which contains the reconstruction results from `dgmmvangerven.py`.  

## Miyawaki Dataset  

To perform reconstruction on the Miyawaki dataset, execute:  

```sh
runmy.bat
```  

This script will run the Python scripts `dgmmmiyawaki.py` and `fid.py`.  

- **`dgmmmiyawaki.py`**: Performs image reconstruction with MSE error, generating an output folder for each hyperparameter, which contains subfolders with the reconstruction results.  
- **`fidmy.py`**: Calculates the FID from the `stim` folder, which contains the original stimuli, and the `rec` folder, which contains the reconstruction results from `dgmmmiyawaki.py`.  
