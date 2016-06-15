import os, sys, csv

def get_data(fname):
    df = []
    with open (fname) as f:
        reader = csv.reader(f)
        for row in reader:
            df.append(row[0].split())
    return df

if len(sys.argv) != 2:
    print("Usage: python read_obs.py <path-to-obs-csv>")
    exit()

fname = sys.argv[1]
get_data(fname)


