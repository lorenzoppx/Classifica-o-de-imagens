from os import listdir
from os.path import isfile, join
mypath="28-05-2021_10-34-24_B68715"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print(onlyfiles)