import os, sys


def get_color_dic(cwd):
    dic_model = {}
    model_color_n = 0
    for f in os.listdir(cwd):
        if f.endswith(".nc") or f.endswith("tif"):
            model = f.split("_", 1)[0].split("-",1)[0]
            if model not in dic_model:
                dic_model[model] = model_color_n
                model_color_n += 1

    return dic_model


if len(sys.argv) != 2:
    print("Usage: python catagory.py <netCDF dir>")

cwd = sys.argv[1]

get_color_dic(cwd)
