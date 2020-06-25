"""
eval_cmd.py
Provides command line interface to evaluate networks on isa
Copyright (C) 2020, Akhilan Boopathy <akhilan@mit.edu>
                    Sijia Liu <Sijia.Liu@ibm.com>
                    Gaoyuan Zhang <Gaoyuan.Zhang@ibm.com>
                    Cynthia Liu <cynliu98@mit.edu>
                    Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
                    Shiyu Chang <Shiyu.Chang@ibm.com>
                    Luca Daniel <luca@mit.edu>
"""

import os
import tensorflow as tf
import funcs
import logging

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_enum(
    "dataset", None, ['mnist', 'cifar', 'tinyimagenet', 'restimagenet'],
    "dataset")

flags.DEFINE_enum(
    "num_class", None, ['one_class', 'two_class', 'all_class'],
    "num_class")

flags.DEFINE_integer(
    "num_image", None,
    "num_image")

flags.DEFINE_enum(
    "norm", None, ['l1', 'l2'],
    "norm")

flags.DEFINE_enum(
    "network_type", None, ['small', 'pool', 'wresnet'],
    "network_type")

flags.DEFINE_bool(
    "rep_l", None,
    "repl")
flags.DEFINE_enum("inp_method", None, ['cam', 'cam++', 'ig'], "inp_method")
flags.DEFINE_string("network", None, "network")
flags.DEFINE_string("output_dir", None, "output_dir")
flags.DEFINE_list("penalties", None, "penalties")
flags.DEFINE_string("task", None, "task")


# ["isa_cam", "tradeoff_cam", "tradeoff_ig", "tradeoff_repr", "isa_ig", "attack_and_display"]
def main(_):
    dataset = FLAGS.dataset
    num_class = FLAGS.num_class
    norm = FLAGS.norm
    inp_method = FLAGS.inp_method
    rep_l = FLAGS.rep_l
    network_type = FLAGS.network_type
    network = FLAGS.network
    penalties = [float(i) for i in FLAGS.penalties]
    num_image = FLAGS.num_image
    kappa = 0.1
    task = FLAGS.task
    tf.logging.info(f"----------task: {task}")
    tf.logging.info(f"----------dataset: {dataset}")
    tf.logging.info(f"----------inp_method: {inp_method}")
    tf.logging.info(f"----------num_class: {num_class}")
    tf.logging.info(f"----------norm: {norm}")
    output_file = os.path.join(FLAGS.output_dir, f"{norm}_{num_class}_{task}_{dataset}_{inp_method}.tsv")
    with tf.Session() as sess:
        result = getattr(funcs, task)(network, sess, network_type, penalties, dataset, num_class, norm, inp_method,
                                      rep_l, kappa, num_image, output_file)
        print(result)
    pass


def setup_env():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    tf.logging.set_verbosity(tf.logging.INFO)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.propagate = False


if __name__ == "__main__":
    flags.mark_flag_as_required("dataset")
    flags.mark_flag_as_required("num_class")
    flags.mark_flag_as_required("num_image")
    flags.mark_flag_as_required("norm")
    flags.mark_flag_as_required("inp_method")
    flags.mark_flag_as_required("rep_l")
    flags.mark_flag_as_required("network_type")
    flags.mark_flag_as_required("network")
    flags.mark_flag_as_required("penalties")
    flags.mark_flag_as_required("task")
    flags.mark_flag_as_required("output_dir")

    tf.gfile.MakeDirs(FLAGS.output_dir)

    setup_env()
    tf.app.run()
    pass
