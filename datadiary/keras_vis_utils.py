"""Utilities related to model visualization."""

# Copied from Keras 2.2.4 distribution
# Modified so that dpi is an argument.
# Modified so that background transparency is an option.
# This file will be unnecessary once the next version of Keras is released
#
# COPYRIGHT
#
# All contributions by François Chollet:
# Copyright (c) 2015 - 2019, François Chollet.
# All rights reserved.
#
# All contributions by Google:
# Copyright (c) 2015 - 2019, Google, Inc.
# All rights reserved.
#
# All contributions by Microsoft:
# Copyright (c) 2017 - 2019, Microsoft, Inc.
# All rights reserved.
#
# All other contributions:
# Copyright (c) 2015 - 2019, the respective contributors.
# All rights reserved.
#
# Each contributor holds copyright over their respective contributions.
# The project versioning (Git) records all such contribution source information.
#
# LICENSE
#
# The MIT License (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from contextlib import redirect_stderr

# Mute Tensorflow chatter messages ('1' means filter out INFO messages.)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# Mute Keras chatter messages
with redirect_stderr(open(os.devnull, "w")):
    from keras.layers.wrappers import Wrapper
    from keras.models import Sequential

# `pydot` is an optional dependency,
# see `extras_require` in `setup.py`.
try:
    import pydot
except ImportError:
    pydot = None


def _check_pydot():
    """Raise errors if `pydot` or GraphViz unavailable."""
    if pydot is None:
        raise ImportError(
            'Failed to import `pydot`. '
            'Please install `pydot`. '
            'For example with `pip install pydot`.')
    try:
        # Attempt to create an image of a blank graph
        # to check the pydot/graphviz installation.
        pydot.Dot.create(pydot.Dot())
    except OSError:
        raise OSError(
            '`pydot` failed to call GraphViz.'
            'Please install GraphViz (https://www.graphviz.org/) '
            'and ensure that its executables are in the $PATH.')


def model_to_dot(model,
                 show_shapes=False,
                 show_layer_names=True,
                 rankdir='TB',
                 dpi=96,
                 transparent_bg=False,
                 ):
    """Convert a Keras model to dot format.

    # Arguments
        model: A Keras model instance.
        show_shapes: whether to display shape information.
        show_layer_names: whether to display layer names.
        rankdir: `rankdir` argument passed to PyDot,
            a string specifying the format of the plot:
            'TB' creates a vertical plot;
            'LR' creates a horizontal plot.

    # Returns
        A `pydot.Dot` instance representing the Keras model.
    """
    _check_pydot()
    dot = pydot.Dot()
    dot.set('rankdir', rankdir)
    if transparent_bg:
        dot.set('bgcolor', "#ffffff00")
    dot.set('concentrate', True)
    dot.set('dpi', dpi)
    dot.set_node_defaults(shape='record')

    if isinstance(model, Sequential):
        if not model.built:
            model.build()
    layers = model.layers

    # Create graph nodes.
    for layer in layers:
        layer_id = str(id(layer))

        # Append a wrapped layer's label to node's label, if it exists.
        layer_name = layer.name
        class_name = layer.__class__.__name__
        if isinstance(layer, Wrapper):
            layer_name = '{}({})'.format(layer_name, layer.layer.name)
            child_class_name = layer.layer.__class__.__name__
            class_name = '{}({})'.format(class_name, child_class_name)

        # Create node's label.
        if show_layer_names:
            label = '{}: {}'.format(layer_name, class_name)
        else:
            label = class_name

        # Rebuild the label as a table including input/output shapes.
        if show_shapes:
            try:
                outputlabels = str(layer.output_shape)
            except AttributeError:
                outputlabels = 'multiple'
            if hasattr(layer, 'input_shape'):
                inputlabels = str(layer.input_shape)
            elif hasattr(layer, 'input_shapes'):
                inputlabels = ', '.join(
                    [str(ishape) for ishape in layer.input_shapes])
            else:
                inputlabels = 'multiple'
            label = '%s\n|{input:|output:}|{{%s}|{%s}}' % (label,
                                                           inputlabels,
                                                           outputlabels)
        if transparent_bg:
            node_kwargs = {'style':'filled', 'fillcolor':'#ffffffff'}
        else:
            node_kwargs = {}
        node = pydot.Node(layer_id, label=label, **node_kwargs)
        dot.add_node(node)

    # Connect nodes with edges.
    for layer in layers:
        layer_id = str(id(layer))
        for i, node in enumerate(layer._inbound_nodes):
            node_key = layer.name + '_ib-' + str(i)
            if node_key in model._network_nodes:
                for inbound_layer in node.inbound_layers:
                    inbound_layer_id = str(id(inbound_layer))
                    # add node if inbound_layer_id node is not present
                    if not dot.get_node(inbound_layer_id):
                        dot.add_node(
                                pydot.Node(
                                    inbound_layer_id,
                                    label='Input',
                                    **node_kwargs
                                    )
                                )
                    dot.add_edge(pydot.Edge(inbound_layer_id, layer_id))
    return dot


def plot_model(model,
               to_file='model.png',
               show_shapes=False,
               show_layer_names=True,
               rankdir='TB',
               dpi=96,
               transparent_bg=False
               ):
    """Converts a Keras model to dot format and save to a file.

    # Arguments
        model: A Keras model instance
        to_file: File name of the plot image.
        show_shapes: whether to display shape information.
        show_layer_names: whether to display layer names.
        rankdir: `rankdir` argument passed to PyDot,
            a string specifying the format of the plot:
            'TB' creates a vertical plot;
            'LR' creates a horizontal plot.
    """
    dot = model_to_dot(
            model, show_shapes, show_layer_names, rankdir, dpi, transparent_bg
            )
    _, extension = os.path.splitext(to_file)
    if not extension:
        extension = 'png'
    else:
        extension = extension[1:]
    dot.write(to_file, format=extension)
