import numpy as np
from scipy.stats import multivariate_normal

def get_snippet_mean(image_snippet, kernel_size, centre):

    snippet_sum = np.sum(image_snippet)

    sum_x = 0
    sum_y = 0
    
    if snippet_sum == 0:
        return np.zeros(2), snippet_sum

    xrange = range(kernel_size) - centre[0];
    yrange = range(kernel_size -1, -1, -1) - centre[1]

    sum_x = np.sum(image_snippet * np.tile(xrange,[kernel_size,1]))
    sum_y = np.sum(image_snippet.T * np.tile(yrange,[kernel_size,1]))

    mean = np.array([sum_x,sum_y])/snippet_sum

    return mean, snippet_sum
    
    
def estimate_covariance(image_snippet, snippet_sum, kernel_size, mean):

    xrange = np.array(range(kernel_size) - mean[0]);
    yrange = np.flip(np.array(range(kernel_size) - mean[1]));

    temp_vec = np.zeros((2,2,image_snippet.ravel().shape[0]))
    
    temp_vec[0,0,:] = np.repeat(xrange*xrange,kernel_size).ravel() * image_snippet.T.ravel()
    temp_vec[0,1,:] = np.outer(xrange,yrange).ravel() * image_snippet.T.ravel()
    temp_vec[1,1,:] = np.tile(yrange*yrange,[kernel_size,1]).ravel() * image_snippet.T.ravel()
    temp_vec[1,0,:] = np.outer(xrange,yrange).ravel() * image_snippet.T.ravel()

    cov_matrix = np.sum(temp_vec,2)/snippet_sum

    return cov_matrix


def get_aniso_gauss(cov_matrix, pos):
    rv = multivariate_normal([0.0,0.0], cov_matrix)
    return rv.pdf(pos)


def kde_image (image , kernel_size , scale ):
    new_img = np. zeros ( shape = image .shape , dtype = np. float32 )
    start_index = np. floor ( kernel_size /2). astype (int)
    end_index_row = image . shape [0] - start_index
    end_index_col = image . shape [1] - start_index
    centre = np. array ([np. floor ( kernel_size / 2). astype ( int), \
                         np. floor ( kernel_size / 2). astype ( int)])

    y, x = np. mgrid [ start_index :-( start_index +1):-1,- start_index :
                      start_index +1:1]
    pos = np. empty (x. shape + (2 ,))
    pos[:, :, 0] = x;
    pos[:, :, 1] = y
    
    for row in range ( start_index , end_index_row ):
        for column in range ( start_index , end_index_col ):
            if image [row , column ] == 0:
                continue

            image_snippet = image [row- start_index :row+ start_index +1 ,\
                                   column - start_index : column + start_index +1].astype(np.float32)

            image_snippet [0,0] = image_snippet [0,0] + 0.001
            image_snippet [0,-1] = image_snippet [0,-1] + 0.001
            image_snippet [-1,0] = image_snippet [-1,0] + 0.001
            image_snippet [-1,-1] = image_snippet [-1,-1] + 0.001

            mean_point,snippet_sum = get_snippet_mean(image_snippet, kernel_size, centre)

            cov_matrix = estimate_covariance(image_snippet,snippet_sum,kernel_size,mean_point+centre)
            
            eigen = np.linalg.eig(cov_matrix)
            norm = np.linalg.norm(eigen[0])
            
            cov_matrix = cov_matrix / norm
            
            gauss_kernel_norm = get_aniso_gauss(cov_matrix*scale, pos)
            gauss_kernel_norm = gauss_kernel_norm/np.sum(gauss_kernel_norm)
            
            gauss_kernel_norm = np.array(gauss_kernel_norm, dtype = np.float32)

            new_img[row - start_index :row + start_index +1, column - start_index :column + start_index +1] \
            += gauss_kernel_norm * image [row , column ]
            
    return new_img
