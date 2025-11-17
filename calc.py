# This function will return the nth triangular number

def triangular_numbers(n):
    if n < 1 or n != int(n):
        raise ValueError("Must be a positive integer")
    else:
        return n * (n+1) / 2
    
print(triangular_numbers(100))
