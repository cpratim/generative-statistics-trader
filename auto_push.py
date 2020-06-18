import os
import json

def read_data(f):
    with open(f, 'r') as df:
        return json.loads(df.read())

def dump_data(f, d):
    with open(f, 'w') as df:
        json.dump(d, df, indent=4)

os.system('rm -r __pycache__')
excluded = ['data', '__pycache__']
#pushes = read_data('pushes.json')['pushes']
#pushes += 1
for f in os.listdir():
	if f not in excluded:
		c = os.system(f'git add {f}')
os.system(f'git commit -m "Version: 302"')
os.system('git push -f origin master')
#dump_data('pushes.json', {'pushes': pushes})

#os.system()