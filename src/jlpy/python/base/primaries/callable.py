"""
Python Language Primaries - Callable.

If a class defines the __call__() method, its instance calls a callabe object.
Any python expression which points to a callabe class can use obj() format to \
call the __call__() class method.

:Author:  JLDP
:Version: 2023.12.05.1
"""


class TestCallable:
    """
    Python Callable Test Class.

    .. card::
    """

    def __init__(self) -> None:
        """Construct a class instance."""
        self.header = "Hello "
        self.footer = "."

    def __call__(self, s: str) -> None:
        """
        Callable method for class instance.

        :param s: callabe input.
        :type s: str
        :return: No return
        :rtype: None
        """
        print(self.header + s + self.footer)


if __name__ == "__main__":
    tc = TestCallable()
    tc("World")
