load "$NCARG_ROOT/lib/ncarg/nclscripts/esmf/ESMF_regridding.ncl"
;load "./gaussim_func.ncl"	;with gaussian window "GauSSIM"
;load "./diffssim_func.ncl"		;with difference map

begin
  ; get the list of models we're interested in...
  modelsList = "../models.csv"
  models = asciiread(modelsList, -1, "string")
  models := models(1:)   ; skip the header record

  delim = ","
  shortnames = str_strip(str_get_field(models, 1, delim))
  filenames = str_strip(str_get_field(models, 2, delim))
  scores =  tofloat(str_get_field(models, 3, delim))

  ; For now, regrid here. Later, we'll refactor this step out and do only once...
  ;---Regridding options
  Opt                   = True
  Opt@InterpMethod      = "bilinear"
  Opt@DstGridType       = "1deg"
  Opt@ForceOverwrite    = True     ; force weights file to be overwritten

  ;---Reference file & variable
  ;; For SST  
  obs_file = "../HadISST.cvdp_data.1901-2010.nc"  
  select_var="sst_spatialmean_ann"
  ;; For PSL
  ;obs_file = "../20thC_ReanV2.cvdp_data.1901-2010.nc"
  ;select_var = "psl_spatialmean_ann"
  
  print("Regridding observations...")
  fref = addfile(obs_file,"r")
  obs = fref->$select_var$
  obsRegrid = ESMF_regrid(obs,Opt)     ; Do the regridding *******

  ; Now, loop through files, regrid, and compute SSID...
  ;---SSIM parameters
  W=7	;window size (odd number>1)
  max_ssim=1.0
  min_ssim=1.0	;adjust according to different variables
  ssim_steps=100
  ssimIndices = new((/ dimsizes(filenames) /), float)

  do i=0, dimsizes(filenames)-1
    f = addfile("../" + filenames(i), "r")
    print(i + ".  Regridding " + filenames(i))
    varRegrid = ESMF_regrid(f->$select_var$, Opt)

    ssim =DiffSSIM(obsRegrid, varRegrid, W)
    ssimIndices(i) = ssim[0]

    delete([/ f, ssim, varRegrid /])   
  end do    

  ; write the .stf file header...wow this is clumbsy!
  ;stfFile = "totalRanking.stf"
  ;write_table(stfFile, "w", [/ "5" /], "%s")
  ;write_table(stfFile, "a", [/ "CVDP_rank float" /], "%s")
  ;write_table(stfFile, "a", [/ "SSIM_rank float" /], "%s")
  ;write_table(stfFile, "a", [/ "model string" /], "%s")
  ;write_table(stfFile, "a", [/ "score string" /], "%s")
  ;write_table(stfFile, "a", [/ "ssid string" /], "%s")

  ;do i=0, dimsizes(filenames)-1
  ;  write_table(stfFile, "a", [/ scores(i) , ssimIndices(i), shortnames(i), scores(i), ssimIndices(i) /], \
  ;      "%f %f %s %f %f")
  ;end do

end
