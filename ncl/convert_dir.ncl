;; read directory

dir = "/Users/puxiaoadmin/cmip5/"
cmd = "cd " + dir + "; ls *.nc"
f_list = systemfunc(cmd)

n_files = dimsizes(f_list)

;; for f in dir
;; do regridding and output csv
target_dir = dir + "csv-sst/"       ;; TODO

ret = systemfunc("rm -rf " + target_dir)
cmd = "mkdir " + target_dir
ret = systemfunc(cmd)
do i=0, n_files - 1
    nc_file_path = dir + f_list(i) 
    f = addfile(nc_file_path, "r")
    ;printVarSummary(f)
    ;printVarSummary(f->psl_spatialmean_ann)
    ;psl := f->psl_spatialmean_ann

    psl := f->sst_spatialmean_ann       ;; TODO

    LAT = fspan(-80, 80, 80)
    LAT@unit = "degree_north"
    LON = fspan(0, 357.5, 144)
    LON@unit = "degree_east"

    ;; regrid
    PSL = linint2(f->lon, psl&lat, psl, True, LON, LAT, 0)

    ; write to a csv file
    opt = True
    opt@fout = target_dir + f_list(i) + "-sst.csv"      ;; TODO
    write_matrix(PSL, "144f8.2", opt)
end do
