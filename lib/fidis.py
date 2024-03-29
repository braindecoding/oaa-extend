import matplotlib.pyplot as plt
#from pytorch_fid import fid_score

from pytorch_fid import fid_score
import tensorflow as tf

def calculate_fid(real_images, generated_images, batch_size=10):
    gpu=tf.config.list_physical_devices('GPU')
    if len(gpu)==0:
        dev='cpu'
    else:
        dev='cuda'
    fid = fid_score.calculate_fid_given_paths([real_images, generated_images], batch_size=min(batch_size, len(real_images)), device=dev, dims=2048)
    return fid

def save_array_as_image(array, filename):
    plt.imshow(array, cmap='hot')  # You can change the colormap as needed
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    
# def calculate_fid(real_images, generated_images):
#     fid = fid_score.calculate_fid_given_paths([real_images, generated_images], batch_size=50, device='cuda', dims=2048)
#     return fid