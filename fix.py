import os, glob
for f in glob.glob('*.py'):
    if f == 'fix.py': continue
    with open(f, 'r') as file:
        c = file.read()
    c = c.replace('\\\"', '\"').replace('\\\\n', '\\n')
    with open(f, 'w') as file:
        file.write(c)
