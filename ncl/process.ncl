;; ======  open and read dir =====

;cwd = "/Users/puxiaoadmin/cmip5/"

;f_list = systemfunc("cd " + cwd"; ls ACCESS1* ")

; TODO passing in cmd line arguments?

;; ========= regridding ==========
; if rectilinear

; systemfunc()



fn = "MRI-CGCM3_19.cvdp_data.1900-2005.nc"
dir = "/Users/puxiaoadmin/cmip5/"
f = addfile(dir + fn, "r")
PSL = f->psl_trends_ann
MAM = f->psl_trends_mam
printVarSummary(PSL)
printVarSummary(PSL&lat)
;print(min(PSL))
;print(num(ismissing(PSL)))





;; ========= visualization ==========
;; step 1: gsn: getting started using ncl
;;         csm: climate system model
;; _FillValue ignored, missing_value not ignored

;; step 1.5: get data


;; step 2: open a Workstation
wks = gsn_open_wks("x11", "test")
res = True
;; step 3: set resources
;res@cnFillOn = True
;res@cnLinesOn = False
;res@mpGridAndLimbOn = True
;res@mpGridLineColor = "grey"
;res@mpGridLineDashPattern = 2

res@mpProjection  = "Robinson"
res@tiMainString = fn

;res@mpMinLonF = -120
;res@mpMaxLonF = -60

;res@cnFillMode = "RasterFill"   ; faster?
;res@cnRasterSmoothingOn = True  ; smoother



;; --------panel-ing
res@gsnDraw = False
res@gsnDraw = False

plot = gsn_csm_contour_map(wks, PSL, res)
mam = gsn_csm_contour_map(wks, MAM, res)
gsn_panel(wks, (/plot, mam/), (/2,1/), False)
;; ------ end panel

;tres = True
;tid = gsn_add_text(wks, plot, "Boulder", -104, 40, tres)
;mkres = True
;mkres@gsMarkerIndex = 17    ;; filled dot
;mkres@gsMarkerSizeF = 0.04
;mkres@gsMarkerColor = "HotPink"

;mid = gsn_add_polymarker(wks, plot, -104, 40, mkres)
;draw(plot)      ; will also draw the text
;frame(wks)      ; advance the frame

