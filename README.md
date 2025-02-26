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


 
