class MultipleOptimizer(object):
    def __init__(self, opt_cae, opt_cnn):
        self.opt_cae = opt_cae
        self.opt_cnn = opt_cnn

    def __str__(self):
        return [opt.__str__() for opt in [self.opt_cae, self.opt_cnn]]

    def zero_grad(self):
        self.opt_cae.zero_grad()
        self.opt_cnn.zero_grad()

    # def step(self, cur_epoch):
    #     self.opt_cae.step()
    #     if self.train_cnn_from >= cur_epoch:
    #         self.opt_cnn.step()
