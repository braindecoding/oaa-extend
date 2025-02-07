# OAA

High-Quality Reconstruction of fMRI Visual Stimuli Using Optimized Autoencoder Architecture

Perhitungan fid harus terpisah karena dia memanfaatkan multi threading jadi tidak bisa disatukan dalam satu file

## Dataset Van Gerven

Untuk melakukan uji dengan dataset Van Gerven run:

```sh
runvg.bat
```

Script ini akan melakukan run script python dgmmvangerven dan fidvg

* dgmmvangerven.py : untuk melakukan rekonstruksi citra dengan error MSE, mengeluarkan folder output setiap hyperparameter yang didalam nya terdapat sub folder hasil rekon
* fidvg.py : melakukan perhitungan FID dari folder stim yang berisi stimulus asal dan rec yang merupakan hasil rekonstruksi dari file dgmmvangerven.py

## Dataset Miyawaki

Untuk melakukan rekonstruksi pada data set miyawaki:

```sh
runmy.bat
```

Script ini akan melakukan run script python dgmmmiyawaki.py dan fid

* dgmmmiyawaki.py : untuk melakukan rekonstruksi citra dengan error MSE, mengeluarkan folder output setiap hyperparameter yang didalam nya terdapat sub folder hasil rekon
* fidvg.py : melakukan perhitungan FID dari folder stim yang berisi stimulus asal dan rec yang merupakan hasil rekonstruksi dari file dgmmmiyawaki.py
