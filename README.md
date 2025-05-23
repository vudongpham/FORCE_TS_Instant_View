# (PROTOTYPE) FORCE Time-series Instant View (FTIV) 
- Instant visualize time-series data of a given location from [FORCE](https://github.com/davidfrantz/force) level-2 datacube.
- Instant visualize multiple data aggregation, interpolation methods
- No external data is written, space saved!
## &#8594; [Here is the example result after running the tool](https://vudongpham.github.io/FORCE_TS_Instant_View)

### 1. Installation
Install with pip
```
python -m pip install git+https://github.com/vudongpham/FORCE_TS_Instant_View.git
```
Using docker image
```
docker pull vudongpham/ftiv:latest
```
Using singularity
```
singularity pull -F ~/ftiv_latest.sif docker://vudongpham/ftiv:latest
```
### 2. Example run
Run with python environment
```
ftiv --daterange 20180101,20191231 \
    --sensor all \
    --band NIR \
    /path/to/your/datacube/level2 \
    52.356415,13.369327
```
Run with docker image
```
docker run --rm  \
    -v $(pwd):$(pwd) \
    -w $(pwd) \
    -v /path/to/your/datacube/level2:/level2dir \
    vudongpham/ftiv ftiv \
    --daterange 20180101,20191231 \
    --sensor all \
    --band NIR \
    /level2dir \
    52.356415,13.369327
```
Run with singularity
```
singularity exec ~/ftiv_latest.sif \
    ftiv --daterange 20180101,20191231 \
    --sensor all \
    --band NIR \
    /path/to/your/datacube/level2 \
    52.356415,13.369327
```
<i>Required arguments:</i>

- `level2dir` \
  FORCE datacube Level-2 directory path, the "datacube-definition.prj" file MUST exist in this directory

- `coordinates`\
  Geographic coordinates Lat,Lon (Y,X) separated by ","

> [!IMPORTANT]  
> If the Latitude,Longitude starts with negative value, e.g., `-60.0,50.0`. You must add `--` before `coordinates` \
> E.g., `-- -60.0,50.0.`

<i>Optional arguments:</i>
- `-d` | `--daterange`: Start date and end date = date range to be considered. Valid values: [YYYYMMDD,YYYYMMDD] <br><br>
- `-s` | `--sensors`:   List of sensors separated by ",". Valid values: LND05,LND07,LND08,LND09,SEN2A,SEN2B,... or all. Default: all <br><br>
- `-b` | `--band`:  Which band to view. Valid value: only one from [RED, GREEN, BLUE, NIR, SWIR1, SWIR2, NDVI]. Default: NDVI <br><br>
- `--printarray` : Call this if you only want to print out array lists of spectral values and dates. Creating HTML will be disabled. \
      Dates    : [days since 1970-01-01, ...] \
      Spectral : [spectral value, ...]
- `--bestquality` : Enables cloud masking more aggressivly, resulting better quality observations, but potentially less quantity. Default: False. 
### 3. Result
A HTML file will be created, looks like [this](https://vudongpham.github.io/FORCE_TS_Instant_View).\
(Note: The buttons for `Paper` and `UDF` are just now placeholders!)
 
