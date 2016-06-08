; comment

N = 10
M = 12
x = random_uniform(-1,1,(/N,M/))    ; random number generators

printVarSummary(x)

x!0 = "time"    ; name the dimension, 0-th column
x!1 = "lat"

printVarSummary(x)

x@long_name = "foo"
x@units = "anything"

printVarSummary(x)

; --- create a variable called time
time = ispan(1,N,1)     ; 1,2,3,..,N
time!0 = "time"         ; Name time's only dimension
time@units = "bogus"
time@long_name = "Time description"
print(time)


x&time = time   ; assign time as a coordinate array of x
printVarSummary(x)


; lat
lat = fspan(-90,90,M)
lat!0 = "lat"
lat@units="degree_north"
lat@long_name="Latitude"

x&lat = lat
printVarSummary(x)


; Coordinates don't necessarily imply Coordinate Variables


; write to a NetCDF file
fname = "foo.nc"
system("rm -f " + fname)

fout = addfile(fname, "c")
fout@title = "NCL workshop"
fout@creation_date = systemfunc("date")

; write the variable to the file
fout->X = x ; will be called X on the file

