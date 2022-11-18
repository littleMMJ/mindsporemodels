"""
##############export checkpoint file into air, onnx, mindir models#################
python export.py
"""
import argparse
import numpy as np

from mindspore import dtype as mstype
from mindspore import Tensor, load_checkpoint, load_param_into_net, export, context

from src.models import Dense24
from src.config import config

parser = argparse.ArgumentParser(description='3dcnn_export')
parser.add_argument('--device_id', type=int, default=0, help='Device id.')
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument('--ckpt_file', type=str, required=True, help='Checkpoint file path.')
parser.add_argument('--file_name', type=str, default='dense24', help='Output file name.')
parser.add_argument('--file_format', type=str, choices=["AIR", "ONNX", "MINDIR"], default='MINDIR', help='file format')
parser.add_argument('--device_target', type=str, choices=['Ascend', 'CPU', 'GPU'], default='Ascend',
                    help='Device target')
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, save_graphs=False)
if args.device_target == "Ascend":
    context.set_context(device_id=args.device_id)

if __name__ == '__main__':
    network = Dense24(num_classes=config.num_classes)

    assert args.ckpt_file is not None, "args.ckpt_file is None."
    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(network, param_dict)

    flair_t2_node = Tensor(np.ones([args.batch_size, 2, config.height_size, config.width_size, config.channel_size]),
                           mstype.float32)
    t1_t1ce_node = Tensor(np.ones([args.batch_size, 2, config.height_size, config.width_size, config.channel_size]),
                          mstype.float32)
    inputs = [flair_t2_node, t1_t1ce_node]
    export(network, *inputs, file_name=args.file_name, file_format=args.file_format)
