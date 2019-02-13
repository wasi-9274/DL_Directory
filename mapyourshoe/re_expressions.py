import re

with open("simple_file", 'r') as f:
    content = f.read()
    pattern = re.compile(r'[^"\']')
    match = pattern.finditer(content)
    for m in match:
        print(m)
