import torch
import numpy as np 

def load_shj_stim():

    return torch.tensor([
        ## square small light
        [[
            [.0, .0, .0, .0, .0, .0, .0],

            [.0, .0, .0, .0, .0, .0, .0],
            
            [.0, .0, 1., 1., 1., .0, .0],
            
            [.0, .0, 1., .2, 1., .0, .0],
            
            [.0, .0, 1., 1., 1., .0, .0],
            
            [.0, .0, .0, .0, .0, .0, .0],
            
            [.0, .0, .0, .0, .0, .0, .0],
        ]],

        ## square large light
        [[
            [.0, .0, .0, .0, .0, .0, .0],

            [.0, 1., 1., 1., 1., 1., .0],
            
            [.0, 1., .2, .2, .2, 1., .0],
            
            [.0, 1., .2, .2, .2, 1., .0],
            
            [.0, 1., .2, .2, .2, 1., .0],
            
            [.0, 1., 1., 1., 1., 1., .0],
            
            [.0, .0, .0, .0, .0, .0, .0],
        ]],

        ## square small dark
        [[
            [.0, .0, .0, .0, .0, .0, .0],

            [.0, .0, .0, .0, .0, .0, .0],
            
            [.0, .0, 1., 1., 1., .0, .0],
            
            [.0, .0, 1., .8, 1., .0, .0],
            
            [.0, .0, 1., 1., 1., .0, .0],
            
            [.0, .0, .0, .0, .0, .0, .0],
            
            [.0, .0, .0, .0, .0, .0, .0],
        ]],

        ## square large dark
        [[
            [.0, .0, .0, .0, .0, .0, .0],

            [.0, 1., 1., 1., 1., 1., .0],
            
            [.0, 1., .8, .8, .8, 1., .0],
            
            [.0, 1., .8, .8, .8, 1., .0],
            
            [.0, 1., .8, .8, .8, 1., .0],
            
            [.0, 1., 1., 1., 1., 1., .0],
            
            [.0, .0, .0, .0, .0, .0, .0],
        ]],

        ## diamond small light
        [[
            [.0, .0, .0, .0, .0, .0, .0],

            [.0, .0, .0, .0, .0, .0, .0],
            
            [.0, .0, .0, 1., .0, .0, .0],
            
            [.0, .0, 1., .2, 1., .0, .0],
            
            [.0, .0, .0, 1., .0, .0, .0],
            
            [.0, .0, .0, .0, .0, .0, .0],
            
            [.0, .0, .0, .0, .0, .0, .0],
        ]],

        ## diamond large light
        [[
            [.0, .0, .0, .0, .0, .0, .0],

            [.0, .0, .0, 1., .0, .0, .0],
            
            [.0, .0, 1., .2, 1., .0, .0],
            
            [.0, 1., .2, .2, .2, 1., .0],
            
            [.0, .0, 1., .2, 1., .0, .0],
            
            [.0, .0, .0, 1., .0, .0, .0],
            
            [.0, .0, .0, .0, .0, .0, .0],
        ]],

        ## diamond small dark
        [[
            [.0, .0, .0, .0, .0, .0, .0],

            [.0, .0, .0, .0, .0, .0, .0],
            
            [.0, .0, .0, 1., .0, .0, .0],
            
            [.0, .0, 1., .8, 1., .0, .0],
            
            [.0, .0, .0, 1., .0, .0, .0],
            
            [.0, .0, .0, .0, .0, .0, .0],
            
            [.0, .0, .0, .0, .0, .0, .0],
        ]],

        ## diamond large dark
        [[
            [.0, .0, .0, .0, .0, .0, .0],

            [.0, .0, .0, 1., .0, .0, .0],
            
            [.0, .0, 1., .8, 1., .0, .0],
            
            [.0, 1., .8, .8, .8, 1., .0],
            
            [.0, .0, 1., .8, 1., .0, .0],
            
            [.0, .0, .0, 1., .0, .0, .0],
            
            [.0, .0, .0, .0, .0, .0, .0],
        ]],
    ], dtype = torch.float)


def load_mnist():
    import gzip, struct, array

    def parse_labels(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    train_images = torch.tensor(parse_images('_/mnist/train-images-idx3-ubyte.gz'), dtype = torch.float)
    train_labels = torch.tensor(parse_labels('_/mnist/train-labels-idx1-ubyte.gz'), dtype = torch.float)
    test_images  = torch.tensor(parse_images('_/mnist/t10k-images-idx3-ubyte.gz'), dtype = torch.float)
    test_labels  = torch.tensor(parse_labels('_/mnist/t10k-labels-idx1-ubyte.gz'), dtype = torch.float)

    return train_images, train_labels, test_images, test_labels
