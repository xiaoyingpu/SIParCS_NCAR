import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pandas.tools.plotting import parallel_coordinates
data = pd.read_csv("iris.csv")
plt.figure()
parallel_coordinates(data, "Name")
plt.show()
