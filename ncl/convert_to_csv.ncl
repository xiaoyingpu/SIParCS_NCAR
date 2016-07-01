; read the NetCDF file

file_name = "CSIRO-Mk3-6-0_5.cvdp_data.1900-2005.nc"
dir = "/Users/puxiaoadmin/cmip5/"
f = addfile(dir + file_name, "r")

psl = f->tas_spatialmean_ann

printVarSummary(psl)
; regrid to 144 * 90
LAT = fspan(-88, 88, 88)
LAT@unit = "degree_north"
LON = fspan(0, 357.5, 144)
LON@unit = "degree_east"

; bilinear interp.
PSL = linint2(f->lon, psl&lat, psl, True, LON, LAT, 0)
;printVarSummary(PSL)
;print(PSL(0,:))
; write to a csv file
opt = True
opt@fout = file_name + "tas.csv"
; Dimensions and sizes:[90] x [144] 
write_matrix(PSL, "144f8.2", opt)
