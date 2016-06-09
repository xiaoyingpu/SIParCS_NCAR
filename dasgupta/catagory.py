import os, sys


if len(sys.argv) != 2:
    print("Usage: python catagory.py <netCDF dir>")


dic_model = {}
model_color_n = 0
for f in os.listdir(sys.argv[1]):
    if f.endswith(".nc"):
        model = f.split("_", 1)[0].split("-",1)[0]
        if model not in dic_model:
            dic_model[model] = model_color_n
            model_color_n += 1


for m in sorted(dic_model, key=dic_model.get):
    print(m, dic_model[m])
