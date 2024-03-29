# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:47:17 2024

@author: Rolly Maulana Awangga
"""
import sys
from lib.fidis import calculate_fid

if __name__ == '__main__':
    K=int(sys.argv[1])
    intermediate_dim=int(sys.argv[2])
    batch_size=int(sys.argv[3])
    maxiter=int(sys.argv[4])
    experimentname=str(K)+","+str(intermediate_dim)+"," + str(batch_size)+"," + str(maxiter)
    # Calculate FID - Frechet Inception Distance (FID)
    rootfolder=sys.argv[1]+"_"+sys.argv[2]+"_"+sys.argv[3]+"_"+sys.argv[4]+"/"
    stimulus_folder=rootfolder+"stim"
    reconstructed_folder=rootfolder+"rec"
    fid_value = calculate_fid(stimulus_folder,reconstructed_folder)
    print('FID:', fid_value)
    with open("FID_Results.csv", "a") as myfile:
        myfile.write(experimentname+","+str(fid_value)+"\n")