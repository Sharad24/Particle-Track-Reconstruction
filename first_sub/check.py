import os
os.system("g++ check.cpp")
path = os.path.dirname(__file__)

from subprocess import call
ret = call(['bash', '-c', './a.out ' + path])
if ret:
	print("Successful")
