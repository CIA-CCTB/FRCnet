import numpy as np
import tensorflow as tf

def get_kernel_list(size):
    
    size_half = np.floor(size/2.0).astype(int)
    
    r = np.zeros([size])
    r[:size_half] = np.arange(size_half)+1
    r[size_half:] = np.arange(size_half,0,-1)

    c=np.zeros([size])
    c[:size_half] = np.arange(size_half)+1
    c[size_half:] = np.arange(size_half,0,-1)

    [R,C] = np.meshgrid(r,c)

    help_index = np.round(np.sqrt(R**2+C**2))
    kernel_list = []

    for i in range(1, size_half+1):
        
        new_matrix = np.zeros(shape=[size,size])
        new_matrix[help_index==i]=1
        kernel_list.append(new_matrix)

    return tf.constant(kernel_list, dtype=tf.complex64)
    
    
@tf.function()
@tf.autograph.experimental.do_not_convert
def FRC_loss(i1, i2, kernel_list):

    i1 = tf.cast(i1, dtype = tf.complex64)
    i2 = tf.cast(i2, dtype = tf.complex64)
        
    I1 = tf.signal.fft2d(i1)   
    I2 = tf.signal.fft2d(i2)
    
    A = tf.multiply(I1, tf.math.conj(I2))
    B = tf.multiply(I1, tf.math.conj(I1))   
    C = tf.multiply(I2, tf.math.conj(I2))

    A_val = tf.reduce_mean(tf.multiply(A, kernel_list), axis=(1,2))
    B_val = tf.reduce_mean(tf.multiply(B, kernel_list), axis=(1,2))
    C_val = tf.reduce_mean(tf.multiply(C, kernel_list), axis=(1,2))
 
    res = tf.abs(A_val) / tf.sqrt(tf.abs(tf.multiply(B_val,C_val)))
    
    shape = tf.shape(kernel_list)
        
    loss02 = 1.0 - tf.reduce_sum(res[:100]) / 100.0
    loss05 = 1.0 - tf.reduce_sum(res[:250]) / 250.0
    
    f = tf.argmax(res<1/7)
    
    return res, f, loss02, loss05
    
    
    
def smlm_frc(data, frac = 1.0, pixel_size=5):

    x = data["x [nm]"].to_numpy()
    y = data["y [nm]"].to_numpy()

    xbins = np.arange(x.min(), x.max()+1, pixel_size)
    ybins = np.arange(y.min(), y.max()+1, pixel_size)
    
    df = data.sample(frac=frac)
    
    s1 = df.sample(frac = 0.5)
    s2 = df.drop(s1.index)
    
    x1 = s1["x [nm]"].to_numpy()
    y1 = s1["y [nm]"].to_numpy()

    x2 = s2["x [nm]"].to_numpy()
    y2 = s2["y [nm]"].to_numpy()

    img1,xe,ye = np.histogram2d(y1,x1,bins=(ybins,xbins));
    img2,xe,ye = np.histogram2d(y2,x2,bins=(ybins,xbins));
    
    f_range = 1/pixel_size*np.arange(0.5*img1.shape[0])/img1.shape[0]

    frc,f,loss02, loss05 = FRC_loss(img1,img2,get_kernel_list(img1.shape[0]))

    return frc, f, loss02, loss05
