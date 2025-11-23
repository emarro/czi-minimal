class FlopsCounter:
    def __init__(self):
        self.reset()

    def add_flops(self, flops: float):
        self.flops_used += float(flops)

    def get_flops(self):
        return self.flops_used

    def reset(self):
        self.flops_used = 0.0
