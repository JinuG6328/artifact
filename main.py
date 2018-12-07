import argparse
import os 


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--forward", action="store_true", help="solve the forward problem")
parser.add_argument("-a", "--add_noise", action="store_true", help="add the noise in the observation")
parser.add_argument("-i", "--inverse", action="store_true", help="solve the inverse problem")
args = parser.parse_args()

if args.forward:
    os.system('python poisson_3.py')

if args.add_noise:
    os.system('add_noise.py')

if args.inverse:
    os.system('python poisson_4.py')