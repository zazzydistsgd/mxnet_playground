"""
Train mnist, see more explanation at http://mxnet.io/tutorials/python/mnist.html
"""
import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet, fit
from common.util import download_file
import mxnet as mx
import numpy as np
import gzip, struct
from test_code_helper.helper_function import add_fit_args

def read_data(label, image):
    """
    download and read data into numpy
    """
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    with gzip.open(download_file(base_url+label, os.path.join('data',label))) as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        label = np.fromstring(flbl.read(), dtype=np.int8)
    with gzip.open(download_file(base_url+image, os.path.join('data',image)), 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        image = np.fromstring(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
    return (label, image)

def _get_lr_scheduler(args, kv):
    if 'lr_factor' not in args or args.lr_factor >= 1:
        return (args.lr, None)
    epoch_size = args.num_examples / args.batch_size
    if 'dist' in args.kv_store:
        epoch_size /= kv.num_workers
    begin_epoch = args.load_epoch if args.load_epoch else 0
    step_epochs = [int(l) for l in args.lr_step_epochs.split(',')]
    lr = args.lr
    for s in step_epochs:
        if begin_epoch >= s:
            lr *= args.lr_factor
    if lr != args.lr:
        logging.info('Adjust learning rate to %e for epoch %d' %(lr, begin_epoch))

    steps = [epoch_size * (x-begin_epoch) for x in step_epochs if x-begin_epoch > 0]
    return (lr, mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=args.lr_factor))

def get_lenet_symbol(num_classes=10, add_stn=False):
    '''
    get lenet symbol, and split it into multiple modules wrt layers
    this is only for test
    '''
    data = mx.symbol.Variable('data')
    if(add_stn):
        data = mx.sym.SpatialTransformer(data=data, loc=get_loc(data), target_shape = (28,28),
                                         transform_type="affine", sampler_type="bilinear")
    #module 1:------------------------------------------------------------------------------
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
    tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
    pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    mod1 = mx.mod.Module(pool1, label_names=[], context=mx.context.cpu())
    #module 2:------------------------------------------------------------------------------
    # second conv
    data = mx.symbol.Variable('data')
    conv2 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=50)
    tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
    pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    flatten = mx.symbol.Flatten(data=pool2)
    mod2 = mx.mod.Module(flatten, label_names=[], context=mx.context.cpu())

    #module 3:------------------------------------------------------------------------------
    # first fullc
    data = mx.symbol.Variable('data')
    fc1 = mx.symbol.FullyConnected(data=data, num_hidden=500)
    tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
    mod3 = mx.mod.Module(tanh3, label_names=[], context=mx.context.cpu())

    #module 4:------------------------------------------------------------------------------
    # second fullc
    data = mx.symbol.Variable('data')
    fc2 = mx.symbol.FullyConnected(data=data, num_hidden=num_classes)
    # loss
    lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    mod4 = mx.mod.Module(lenet, context=mx.context.cpu())

    #--------------------------------------------------------------------------------
    # Container module
    #--------------------------------------------------------------------------------
    lenet_mod_seq = mx.mod.SequentialModule()
    lenet_mod_seq.add(mod1, take_labels=False).add(mod2, auto_wiring=True, take_labels=False).add(mod3, auto_wiring=True, take_labels=False).add(mod4, take_labels=True, auto_wiring=True)
    return lenet_mod_seq


def to4d(img):
    """
    reshape to 4D arrays
    """
    return img.reshape(img.shape[0], 1, 28, 28).astype(np.float32)/255

def get_mnist_iter(args, kv):
    """
    create data iterator with NDArrayIter
    """
    (train_lbl, train_img) = read_data(
            'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz')
    (val_lbl, val_img) = read_data(
            't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')
    train = mx.io.NDArrayIter(
        to4d(train_img), train_lbl, args.batch_size, shuffle=True)
    val = mx.io.NDArrayIter(
        to4d(val_img), val_lbl, args.batch_size)
    return (train, val)

def _save_model(args, rank=0):
    if args.model_prefix is None:
        return None
    dst_dir = os.path.dirname(args.model_prefix)
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    return mx.callback.do_checkpoint(args.model_prefix if rank == 0 else "%s-%d" % (
        args.model_prefix, rank))

def fit(args, network, data_loader, **kwargs):
    # kvstore
    kv = mx.kvstore.create(args.kv_store)
    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
#    head = '%(asctime)-15s Node[' + str(0) + '] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('start with arguments %s', args)

    # data iterators
    (train, val) = data_loader(args, kv=kv)

    # save model
    checkpoint = _save_model(args, kv.rank)
#    checkpoint = _save_model(args, 0)

    # learning rate
    lr, lr_scheduler = _get_lr_scheduler(args, kv='local')
    # set optimizer params
    optimizer_params = {
            'learning_rate': lr,
            'momentum' : args.mom,
            'wd' : args.wd,
            'lr_scheduler': lr_scheduler}
    # initialize monitor        
    monitor = mx.mon.Monitor(args.monitor, pattern=".*") if args.monitor > 0 else None
    # setup initializer
    if args.network == 'alexnet':
        # AlexNet will not converge using Xavier
        initializer = mx.init.Normal()
    else:
        initializer = mx.init.Xavier(
            rnd_type='gaussian', factor_type="in", magnitude=2)

    # evaluation metrices
    eval_metrics = ['accuracy']
    if args.top_k > 0:
        eval_metrics.append(mx.metric.create('top_k_accuracy', top_k=args.top_k))

    # callbacks that run after each batch
    batch_end_callbacks = [mx.callback.Speedometer(args.batch_size, args.disp_batches)]
    if 'batch_end_callback' in kwargs:
        cbs = kwargs['batch_end_callback']
        batch_end_callbacks += cbs if isinstance(cbs, list) else [cbs]

    # run training
    network.fit(train,
        begin_epoch        = args.load_epoch if args.load_epoch else 0,
        num_epoch          = args.num_epochs,
        eval_data          = val,
        eval_metric        = eval_metrics,
        kvstore            = kv,
        optimizer          = args.optimizer,
        optimizer_params   = optimizer_params,
        initializer        = initializer,
        batch_end_callback = batch_end_callbacks,
        epoch_end_callback = checkpoint,
        allow_missing      = True,
        monitor            = monitor)

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description="train mnist",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num-classes', type=int, default=10,
                        help='the number of classes')
    parser.add_argument('--num-examples', type=int, default=60000,
                        help='the number of training examples')
    parser.add_argument('--profile_filename', type=str, default='/hwang/home/log_files/json_files/mxnet_symbol_test/single_iter_test.json')
    add_fit_args(parser)
    parser.set_defaults(
        # network
        network        = 'mlp',
        # train
        gpus           = None,
        batch_size     = 128,
        disp_batches   = 1,
        num_epochs     = 10,
        lr             = .05,
        lr_step_epochs = '10',
    )
    args = parser.parse_args()
    # add profiler here to test detailed ops performance
#    mx.profiler.profiler_set_config(mode='all', filename=args.profile_filename)
    # load network
    sym = get_lenet_symbol()

    # train
    fit(args, sym, get_mnist_iter)
