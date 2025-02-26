# (PROTOTYPE) FORCE Time-series Instant View (FTIV) 
- Instant visualize time-series data in a given location from  FORCE level-2 datacube.
- Instant visualize multiple data aggregation, interpolation methods
- No external data is written, space saved!

## [Here is the example result when running the tool](https://vudongpham.github.io/FORCE_TS_Instant_View)

### Installation with pip
```
python -m pip install git+https://github.com/vudongpham/FORCE_TS_Instant_View.git
```

### Example Run
```
ftiv --daterange 20180101,20201231 \
    --sensor all \
    --band NIR \
    /path/to/your/datacube/level2 \
    4547853,3445623
```
<i>Required arguments:</i>

- level2dir \
  FORCE datacube Level-2 directory path, the "datacube-definition.prj" file MUST exist in this directory

- coordinates\
  Projected X,Y coordinates separated by "," (must be the same as the projection defined in datacube-definition.prj) \
  If you provide Y,X format, enable --yx argument


<i>Optional arguments:</i>
- -d | --daterange: Start date and end date = date range to be considered. Valid values: [YYYYMMDD,YYYYMMDD] <br><br>
- -s | --sensors:   List of sensors separated by ",". Valid values: LND05,LND07,LND08,LND09,SEN2A,SEN2B,... or all (Default is all) <br><br>
- -b | --band:  Which band to view. Valid value: only one from [RED, GREEN, BLUE, NIR, SWIR1, SWIR2]. Default: NIR <br><br>
- --yx : Call this if you add coordinate in format Y,X instead of X,Y <br><br>
- --printarray : Call this if you only want to print out array lists of spectral values and dates. Creating report is disabled <br><br>



 
