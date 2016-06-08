;; ======  open and read dir =====

cwd = "/Users/puxiaoadmin/cmip5/"
cmd = "cd " + cwd + "; ls ACCESS1* "
f_list = systemfunc(cmd)

;; ========= regridding ==========

var = "psl_trends_mam"

n_files = dimsizes(f_list)



werkstation = gsn_open_wks("x11", "panel")
res = True

res@gsnDraw = False
res@gsnFrame = False

res@mpMinLonF = -60
res@mpCenterLonF = -20
res@mpMaxLonF = 100
res@mpMinLatF = -55
res@mpMaxLatF = 55

plots = new(4, graphic)

do i=0, n_files-1
    full_path = cwd + f_list(i) ; assuming the extension .nc is there
    f = addfile(full_path, "r")
    ;; TODO what are good grid sizes?
    ;; TODO regional; ENSO, etc.
    psl = f->psl_trends_ann
    ;LAT = fspan(-90, 90, 72)
    ;LAT@unit = "degree_north"
    ;LON = fspan(0, 357.5, 144)
    ;PSL = linint2(f->lon, psl&lat, psl, True, LON, LAT, 0)

    plot = gsn_csm_contour_map(werkstation, psl, res)   ;; TODO: change back to PSL
    if (i.le. 3) then
       plots(i) = plot 
    end if
    ;; TODO visualize a bit?
end do


gsn_panel(werkstation, plots, (/2,2/), False)
