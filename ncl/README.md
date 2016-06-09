```
$ ncl -Q -n 'x=3' y=4 foo.ncl          <- single quote!
Variable: z
Type: integer
Total Size: 4 bytes
            1 values
Number of Dimensions: 1
Dimensions and sizes: [1]
Coordinates:
7

$ cat foo.ncl
z = x+y
print(z)

```
