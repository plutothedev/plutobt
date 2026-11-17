from plutobt.strategy import Strategy


class Parent(Strategy):
    p: int = 1


class Child(Parent):
    c: int = 2
