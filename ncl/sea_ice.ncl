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

    dir = "~/cmip5/sea_ice/"   ;; sea ice!
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
        print(dir + f_list(i))
        f = addfile(dir + f_list(i),"r")

        if (.not.isfilevar(f, "aice_nh_spatialmean_ann")) then
            print("Skipping file (no ice var): " + f_list(i))
	        continue
    	end if
       
        var := f->$select_var$

        Opt@WgtFileName       = "curv_to_1deg.nc"
        if (isfilevar(f, "lat2d_ice_nh")) then
            Opt@SrcGridLat        := f->lat2d_ice_nh
            Opt@SrcGridLon        := f->lon2d_ice_nh ;src_lon
        else if (isfilevar(f, "lat")) then
            Opt@SrcGridLat        := f->lat
            Opt@SrcGridLon        := f->lon
        else if (isfilevar(f, "j")) then
            Opt@SrcGridLat        := f->j
            Opt@SrcGridLon        := f->i
        else
            print("Skipping file (unknown coords): " + f_list(i))
            continue
        end if
        end if
        end if

        Opt@SrcRegional       = False             ;;--Change (maybe)
        ;    Opt@SrcInputFileName  = src_file          ; optional, but good idea
        Opt@SrcMask2D         := where(.not.ismissing(var),1,0) ; Necessary if has
                                                           ; missing values.

        Opt@DstGridType       = "1deg"            ; Destination grid
        Opt@DstLLCorner       = (/-89.75d,   0.00d /)   ;;--Change (likely)
        Opt@DstURCorner       = (/ 89.75d, 359.75d /)   ;;--Change (likely)
        Opt@DstRegional       = False                   ;;--Change (maybe)

        Opt@ForceOverwrite    = True
        Opt@PrintTimings      = False; True
        Opt@Debug             = False; True

        var_regridded := ESMF_regrid(var, Opt)

        ;;; save the variable to file
        new_f_name = f_list(i) + ".reggrided.nc"
        fout := addfile(new_dir + new_f_name, "c")
        fout->$select_var$ = var_regridded
    end do
end
