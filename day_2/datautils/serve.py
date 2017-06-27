
from torch.utils.data import DataLoader
from torch.autograd import Variable


class DataServer(object):
    def __init__(self, vectorized_data):
        self.vectorized_data = vectorized_data
        self.gpu_mode = False
        self.volatile_mode = False

    def serve_batches(self, batch_size, num_batches=-1, num_workers=0):
        datagen = DataLoader(self.vectorized_data, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers)
        for batch_index, batch in enumerate(datagen):
            out = {}
            for key, val in batch.items():
                if not isinstance(val, Variable):
                    val = Variable(val)
                if self.gpu_mode:
                    val = val.cuda()
                if self.volatile_mode:
                    val = val.volatile()
                out[key] = val

            yield out
            if num_batches > 0 and batch_index > num_batches:
                break

    def enable_gpu_mode(self):
        self.gpu_mode = True

    def disable_gpu_mode(self):
        self.gpu_mode = False

