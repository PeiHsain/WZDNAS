# Modify [ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware]
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.
import yaml

class LatencyEstimator(object):
    def __init__(self, table='lookup_table.yaml'):
        with open(table, 'r') as fp:
            self.lut = yaml.load(fp, Loader=yaml.FullLoader)

    @staticmethod
    def repr_shape(shape):
        if isinstance(shape, (list, tuple)):
            return 'x'.join(str(_) for _ in shape)
        elif isinstance(shape, str):
            return shape
        else:
            return TypeError

    def predict(self, stage=None, gamma=None, depth=None):
        """
        latency time: ms
        :param : stage:1-gamma:25-depth:1
            Layer type must be one of the followings
                1. `stage`: The stage in the supernet.
                2. `gamma`: The gamma choice of the search space.
                3. `depth`: The depth choice of the search space.
        """
        # infos = [ltype, 'input:%s' % self.repr_shape(_input), 'output:%s' % self.repr_shape(output), ]
        key = f'stage:{stage}-gamma:{gamma}-depth:{depth}'
        # print(key)

        # if ltype in ('expanded_conv',):
        #     assert None not in (expand, kernel, stride, idskip)
        #     infos += ['expand:%d' % expand, 'kernel:%d' % kernel, 'stride:%d' % stride, 'idskip:%d' % idskip]
        # key = '-'.join(infos)
        return self.lut[key]['mean']