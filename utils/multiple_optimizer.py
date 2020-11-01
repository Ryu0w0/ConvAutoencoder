class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op

    def __str__(self):
        # TODO: 動作確認
        return [opt.__str__() for opt in self.optimizers]

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()
