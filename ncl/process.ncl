;; ======  open and read dir =====

cwd = "/Users/puxiaoadmin/cmip5/"
cmd = "cd " + cwd + "; ls ACCESS1* "
print(cmd)
f_list = systemfunc(cmd)

print(f_list)
;; ========= regridding ==========

n_files = dimsizes(f_list)

do i=0, n_files-1
    full_path = cwd + f_list(i) ; assuming the extension .nc is there
    f = addfile(full_path, "r")
    printVarSummary(f->psl_trends_mam)
end do

