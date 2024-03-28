Rem experimentname=str(K)+"_"+str(intermediate_dim)+"_" + str(batch_size)+"_" + str(maxiter)

Rem indm 128
python dgmmvangerven.py 12 128 10 4000
python fid.py 12 128 10 4000