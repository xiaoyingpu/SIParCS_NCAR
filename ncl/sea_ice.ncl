load "$NCARG_ROOT/lib/ncarg/nclscripts/esmf/ESMF_regridding.ncl"
;; main
begin
    ;; regridding options
    Opt                   = True
    Opt@InterpMethod      = "bilinear"
    Opt@DstGridType       = "1deg"
    Opt@ForceOverwrite    = True     ; force weights file to be overwritten

    ;; target var: sea ice
    select_var= "aice_nh_spatialmean_ann"

    dir = "/Users/puxiaoadmin/cmip5/sea_ice/"   ;; sea ice!
    cmd = "cd " + dir + "; ls *.nc"
    f_list = systemfunc(cmd)
    n_files = dimsizes(f_list)

    print("Regridding members...")
    do i=0, n_files - 1
        f = addfile(f_list(i),"r")
        f_var = f->$select_var$
        var_grid = ESMF_regrid(f_var, Opt)
    end do
end
