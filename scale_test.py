import random


class scale():
    def __init__(self, scales):
        self.scales = scales
    def print_scale(self):
        sc = (self.scales[1] - self.scales[0]) * random.random() - \
       (self.scales[1] - self.scales[0]) / 2 + 1
        print(sc)

scale = scale(scales=(.75, 1.5))
for i in range(50):
    scale.print_scale()
