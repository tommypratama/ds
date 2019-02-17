# Import matlab files using scipy.io
import scipy.io
posts_mat = scipy.io.loadmat('posts-100.mat')
type(posts_mat)
posts_mat.keys()
posts_mat['posts']
