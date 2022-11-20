# %%
from mindspore import nn, ops, Tensor, jit_class, set_context, set_seed
import mindspore as ms
import numpy as np

set_seed(123456)

# %%
set_context(mode=ms.GRAPH_MODE, device_target="GPU")

# %%
from mindspore.common.initializer import Normal
class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        # self.fc1 = nn.Dense(1, 20, Normal())
        # self.fc2 = nn.Dense(20, 20, Normal())
        # self.fc3 = nn.Dense(20, 20, Normal())
        self.fcout = nn.Dense(1, 1, weight_init=Normal(0.02), bias_init=Normal(0.02))
        # self.act = ops.Tanh()

    def construct(self, x):
        # x = self.act(self.fc1(x))
        # x = self.act(self.fc2(x))
        # x = self.act(self.fc3(x))
        x = self.fcout(x)

        return x

model = Network()
model

# %%
class MyIterable:
    def __init__(self):
        samples = 2**17
        self._index = 0
        self._data = np.random.uniform(size=(samples, 1)).astype(np.float32)
        self._label = self._data
        
        
        
    def __next__(self):
        if self._index >= len(self._data):
            raise StopIteration
        else:
            item = (self._data[self._index], self._label[self._index])
            self._index += 1
            return item
        
    def __iter__(self):
        self._index = 0
        return self
    
    def __len__(self):
        return len(self._data)
    
dataset = ms.dataset.GeneratorDataset(source=MyIterable(), column_names=["data", "label"])
print(dataset)

# %%
from matplotlib import pyplot as plt
test_data = np.random.uniform(size=(2**17, 1)).astype(np.float32)
test_label = test_data
plt.scatter(test_data, test_label, s=1)
# plt.show()

# %%
optimizer = nn.SGD(model.trainable_params(), learning_rate=5e-3)
loss_fn = nn.MSELoss()

# %%
dataset = dataset.batch(batch_size=2**5)

# %%
def train_loop(model, dataset, loss_fn, optimizer):
    def forwad_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        # return loss, logits
        return loss

    # grad_fn = ops.value_and_grad(forwad_fn, None, optimizer.parameters, has_aux=True)
    grad_fn = ops.value_and_grad(forwad_fn, None, optimizer.parameters, has_aux=False)
    
    def train_step(data, label):
        # (loss, _), grads = grad_fn(data, label)
        loss, grads = grad_fn(data, label)
        # print("grads: \n", grads)

        loss = ops.depend(loss, optimizer(grads))
        return loss
    
    size = dataset.get_dataset_size()
    print(size)
    model.set_train()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        # print(data.shape, "  ", label.shape)
        loss = train_step(data, label)

        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")
        

# %%
train_loop(model, dataset, loss_fn, optimizer)


