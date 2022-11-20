import mindspore as ms
import mindspore.ops as ops
import numpy as np

import time

ms.set_context(mode=1, device_target="GPU")
time_beg = time.time()
x = ms.Tensor(np.ones([1, 3, 3, 4]).astype(np.float32))
y = ms.Tensor(np.ones([1, 3, 3, 4]).astype(np.float32))
print(ops.add(x, y))
print("time cost: ", time.time() - time_beg)
