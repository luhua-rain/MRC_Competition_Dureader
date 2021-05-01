from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch

from bert.modeling import BertConfig, BertForPreTraining#, load_tf_weights_in_bert
from xlnet.modeling_xlnet import XLNetLMHeadModel, XLNetConfig
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    import re
    import numpy as np
    import tensorflow as tf

    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    # for v in init_vars:
    #     print(v)

    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):
    # Initialise PyTorch model
    config = XLNetConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = XLNetLMHeadModel(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_bert(model, config, tf_checkpoint_path)

    # Save pytorch-model
    print("Save PyTorch model to {}".format(pytorch_dump_path))
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # ## Required parameters
    # parser.add_argument("--tf_checkpoint_path",
    #                     default = './roberta_large/',
    #                     type = str,
    #                     required = True,
    #                     help = "Path to the TensorFlow checkpoint path.")
    # parser.add_argument("--bert_config_file",
    #                     default = './roberta/bert_config_large.json',
    #                     type = str,
    #                     required = True,
    #                     help = "The config json file corresponding to the pre-trained BERT model. \n"
    #                         "This specifies the model architecture.")
    # parser.add_argument("--pytorch_dump_path",
    #                     default = './roberta.bin',
    #                     type = str,
    #                     required = True,
    #                     help = "Path to the output PyTorch model.")
    # args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch('./xlnet_large/',  #args.tf_checkpoint_path,
                                     './xlnet_large/xlnet_config.json',#args.bert_config_file,
                                     './xlnet.bin')  #args.pytorch_dump_path)