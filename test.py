# Test class inheritance to see if I can refactor labeled.py to make it less wet

class Base():
    
    def __init__(self, variable) -> None:
        self.variable = variable
    
    def method(self):
        print(self.variable)

class Inheritor(Base):
    def __init__(self, variable):
        super().__init__(variable)  # Call the superclass's __init__ method to initialize variable


if __name__ == '__main__':
    
    base = Base(variable='Hello')
    base.method()
    
    inheritor = Inheritor(variable='Bye')
    inheritor.method()