; read the NetCDF file

file_name = "20thC_ReanV2.cvdp_data.1901-2010.nc"
dir = "/Users/puxiaoadmin/cmip5_obs/"
f = addfile(dir + file_name, "r")

psl = f->psl_spatialmean_ann

; regrid to 144 * 90
LAT = fspan(-90, 90, 90)
LAT@unit = "degree_north"
LON = fspan(0, 357.5, 144)
LON@unit = "degree_east"

; bilinear interp.
PSL = linint2(f->lon, psl&lat, psl, True, LON, LAT, 0)
printVarSummary(PSL)

; write to a csv file
opt = True
opt@fout = "20thC_ReanV2.cvdp_data.1901-2010.nc.csv"
; Dimensions and sizes:[90] x [144] 
write_matrix(PSL, "144f8.2", opt)
