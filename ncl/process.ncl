;; ======  open and read dir =====

cwd = "/Users/puxiaoadmin/cmip5/"
cmd = "cd " + cwd + "; ls ACCESS1* "
f_list = systemfunc(cmd)

;; ========= regridding ==========

var = "psl_trends_mam"

n_files = dimsizes(f_list)

do i=0, n_files-1
    full_path = cwd + f_list(i) ; assuming the extension .nc is there
    f = addfile(full_path, "r")
    print("--------- before -------")
    printVarSummary(f->$var$)
    psl = f->psl_trends_ann
    LAT = fspan(-90, 90, 72)
    LON = fspan(0, 357.5, 144)
    PSL = linint2(f->lon, psl&lat, psl, True, LON, LAT, 0)
    print("--------- after --------")
    printVarSummary(PSL)

    print("================")

end do

