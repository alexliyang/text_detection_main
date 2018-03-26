# network中方法的专用装饰器
def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.inputs) == 1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.

        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated



class base_network(object):
    def __init__(self):
        pass

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    """载入与训练模型"""

    def load(self):
        pass

