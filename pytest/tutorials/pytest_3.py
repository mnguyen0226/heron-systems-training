# script for assert that certain exceptions is raised
import pytest

def f():
    raise SystemExit(1)

def test_mytest():
    with pytest.raises(SystemExit):
        f()
#------------------------------------
# quick example of hasattr() 
# hasattr() method returns true if an object has the given named attribute and false if it does not
class Hokie:
    age = 22
    name = "Minh"

def validate_hokie():
    person = Hokie()
    print(f"Hokie has age attribute in the class?: {hasattr(person, 'age')}")
    print(f"Hokie has salary attribute in the class?: {hasattr(person, 'salary')}")

#------------------------------------
# Group multiple tests in a class
class TestClass:
    def test_include(self):
        x = "bruh"
        assert "h" in x

    def test_att(self):
        x = "check"
        assert hasattr(x, "check") # pytest will test where we have assert

#------------------------------------
# Having each test share the same class instance would be very detrimental to test isolation and would promote poor test practice
class TestClassDemoInstance:
    def test_one(self):
        assert 1
    
    def test_two(self):
        assert 1



if __name__ == "__main__":
    test_mytest()
    validate_hokie()