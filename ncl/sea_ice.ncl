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
    
    ;; get a new dir to place regridded files
    new_dir = "~/cmip5/sea_ice/regridded/"
    ; del. old one, creat., change dir
    cmd := "rm -rf " + new_dir + "; mkdir " + new_dir 
    ret = systemfunc(cmd)

    print("Regridding members...")
    do i=0, n_files - 1
        print(f_list(i))

        f = addfile(dir + f_list(i),"r")
        ;f_var := f->$select_var$
        foo = ESMF_regrid(f->$select_var$, Opt)

        ;;; save the variable to file
        ;new_f_name = f_list(i) + ".reggrided.nc"
        ;fout = addfile(new_dir + new_f_name, "c")
        ;fout->$select_var$ = (/var_grid/)
    end do
end
