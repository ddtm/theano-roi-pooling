import numpy as np
import theano
import theano.tensor as T

from roi_pooling import ROIPoolingOp

op = ROIPoolingOp(pooled_h=2, pooled_w=2, spatial_scale=1.0)

t_data = T.ftensor4()
t_rois = T.fmatrix()

t_outs = op(t_data, t_rois)
t_c = t_outs[0].sum()

t_g_data = T.grad(t_c, t_data)[0]

f = theano.function([t_data, t_rois], t_outs + [t_g_data])

# Perform actual test.
data = np.random.rand(1, 2, 32, 32).astype(np.single)
rois = np.array([[0, 0, 0, 3, 3],
                 [0, 0, 0, 7, 7]]).astype(np.single)

outs = f(data, rois)

# xv = np.ones((4, 5), dtype="float32")
# assert np.allclose(f(xv), xv * 2)  
# print(np.asarray(f(xv)))
