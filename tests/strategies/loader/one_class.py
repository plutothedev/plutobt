from plutobt.strategy import Strategy


class Only(Strategy):
    length: int = 5
    scale: float = 1.5
    on: bool = True
    label: str = "x"
    _hidden: int = 3

    @property
    def prop(self):
        return 1

    def helper(self):
        return 2
