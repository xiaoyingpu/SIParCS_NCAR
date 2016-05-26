from pylab import scatter,text,show,cm,figure
from pylab import subplot,imshow,NullLocator
from sklearn import manifold, datasets
import os, sys, cv2


f_list = []
X = []

for f in os.listdir(sys.argv[1]):    # img dir as commandline arg
    f_list.append(f)

# change working directory
os.chdir(sys.argv[1])

for f in f_list:
    img = cv2.imread(f)
    X.append(list(img.flat))

print("Length of dataset: {}".format(len(X)))

# load the digits dataset
# 901 samples, about 180 samples per class
# the digits represented 0,1,2,3,4
#digits = datasets.load_digits(n_class=5)
#X = digits.data
#color = digits.target

# running Isomap
# 5 neighbours will be considered and reduction on a 2d space
Y = manifold.Isomap(3, 2).fit_transform(X)

# plotting the result
figure(1)
scatter(Y[:,0], Y[:,1], c='k', alpha=0.3, s=10)
#for i in range(Y.shape[0]):
# text(Y[i, 0], Y[i, 1], str(color[i]),
#      color=cm.Dark2(color[i] / 5.),
#      fontdict={'weight': 'bold', 'size': 11})
show()
