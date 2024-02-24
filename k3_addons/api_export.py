import namex


class k3_export(namex.export):
    def __init__(self, path):
        super().__init__(package="k3_addons", path=path)
