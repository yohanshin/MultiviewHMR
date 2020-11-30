from cfg.defaults import _C as _cfg
from yacs.config import CfgNode as CN

import argparse


def get_cmd():
    parser = argparse.ArgumentParser(description="MarkerlessMoCap training configuration")
    parser.add_argument(
        "--config-file", default="configs/default_configs.yaml", 
        metavar="FILE", help="path to config file")
    parser.add_argument("--image", default="", help="path to demo image file")
    parser.add_argument("--output", default="", help="path to demo output file")
    parser.add_argument(
        "opts", default=None, nargs=argparse.REMAINDER,
        help="Modify config options using the command-line")
    
    return parser
    

def get_cfg(args):
    """
    Define configuration.
    """
    
    if args.config_file is not "":
        _cfg.merge_from_file(args.config_file)
    _cfg.merge_from_list(args.opts)
    _cfg.freeze()

    return _cfg, args