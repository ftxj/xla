import yaml

def search(list, platform):
    for i in range(len(list)):
        if list[i] == platform:
            return True
    return False

with open("xla_native_functions.yaml", "r") as stream:
    try:
        xla_yaml = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
    print(type(xla_yaml))

xla_functions = xla_yaml['supported']
xla_functions.sort()


with open("xla_pytorch_diff.yaml", "r") as stream:
    try:
        totch_yaml = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

torch_functions = []
xla_not_include = []

for func in totch_yaml:
    func_str = func['func']
    func_name = func_str.split('(')[0]
    torch_functions.append(func_name)
    if(search(xla_functions, func_name) == False):
        xla_not_include.append(func_name)
    else:
        print(func_name)    
