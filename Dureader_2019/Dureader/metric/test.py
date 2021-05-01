import json

with open("./ref.json", 'r', encoding='utf-8') as f:
    for line in f.readlines():
        source = json.loads(line.strip())
print("######################## end #################################")
    
