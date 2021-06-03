import pytest
from decimal import Decimal

print(format(2**(1/2), '.60g'))
num = Decimal(2**(1/2))
print(num)

a = 123

def change_a():
    global a
    a = 456

class Test():
    print ("Hello")
    global a
    print(a)
    def test_one(self):
        assert 1
    