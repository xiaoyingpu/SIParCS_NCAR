

dir = "/Users/puxiaoadmin/cmip5/timeseries/"
f_NASA = "seaice_conc_monthly_nh_NASA_Bootstrap_v2.nsidc.v02r00.197811-201412.nc"
f_Walsh = "walsh_chapman.NH.seaice.187001-201112.nc"

f = addfile(dir + f_NASA, "r")

ice = f->SIC

N = dimsizes(ice(:,0,0))    ; # of timestamps

do i=0, N-1
   ;; is using python better? 

end do







