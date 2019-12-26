import torch
import numpy

def main():
    print("working")
    env = numpy.zeros((6, 5, 3))

    env[0, 0, 0] = 99
    env[0, 0, 0] = 66

    print(env)
    print(type(env))

if __name__ == '__main__':
     main()