import rasterio
import numpy as np
from itertools import product
import os
from datetime import datetime, timedelta
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import io
import base64
import os
import argparse
import re
import warnings
warnings.simplefilter("ignore")

# Checking input data
class argparseCondition():
    def dateRangeInput(self, value):
        def is_valid_date(date_str):
            """Check if the input string is a valid date in YYYYMMDD format."""
            try:
                datetime.strptime(date_str, "%Y%m%d")
                return True
            except ValueError:
                return False
        def format_date(a):
            return f'{a[0:4]}-{a[4:6]}-{a[6:8]}'
        
        pattern = r'^\d{8},\d{8}$'
        if not re.match(pattern, value):
            raise argparse.ArgumentTypeError(f"Invalid date range format: {value}. Expected format is YYYYMMDD,YYYYMMDD.")
        start_date, end_date = value.split(',')
        if not is_valid_date(start_date):
            raise argparse.ArgumentTypeError(f"Invalid date {start_date}")
        if not is_valid_date(end_date):
            raise argparse.ArgumentTypeError(f"Invalid date {end_date}")
        return start_date, end_date
    



class FigureBuilder():
    def __init__(self):
        pass

    def spectral_temporal_metrics(self, y):
        return np.percentile(y, [0, 25, 50, 75, 100])

    def linear_interpolation(self, x, y, xtest):
        ytest = np.interp(xtest, x, y)
        return xtest, ytest

    def moving_average(self, x, y, xtest, window_size=15):
        ytest = np.interp(xtest, x, y)
        kernel = np.ones(window_size) / window_size  # Create a kernel for moving average
        ytest = np.convolve(ytest, kernel, mode='same')
        return xtest, ytest
    
    def savitzky_golay_interpolation(self, x, y, xtest,window_size=15, polyorder = 3):

        y_test = savgol_filter(y, window_size, polyorder)
        interpolator_func = interp1d(x, y_test, kind='linear', fill_value='extrapolate')
        ytest = interpolator_func(xtest)
        return xtest, ytest

    def harmonic_function(self, x, y, xtest):
        def objective_simple(x, a0, a1, b1, c1):
            return a0 + a1 * np.cos(2 * np.pi / 365 * x) + b1 * np.sin(2 * np.pi / 365 * x) + c1 * x

        def objective_advanced(x, a0, a1, b1, c1, a2, b2):
            return objective_simple(x, a0, a1, b1, c1) + a2 * np.cos(4 * np.pi / 365 * x) + b2 * np.sin(4 * np.pi / 365 * x)

        def objective_full(x, a0, a1, b1, c1, a2, b2, a3, b3):
            return objective_advanced(x, a0, a1, b1, c1, a2, b2) + a3 * np.cos(6 * np.pi / 365 * x) + b3 * np.sin(
                6 * np.pi / 365 * x)
        
        xtrain = np.concatenate([x - 365, x, x + 365])
        ytrain = np.concatenate([y, y, y])
        func_ = objective_full
        popt, _ = curve_fit(func_, xtrain, ytrain)
        ytest = func_(xtest, *popt)
        return xtest, ytest
    
    def generate_html(self, x, y, xtest, coord_x, coord_y, tile, start_time, end_time, sensor, band):


        func_dict = {
            'Raw data': 0,
            'Spectral-temporal metrics': self.spectral_temporal_metrics,
            'Linear interpolation': self.linear_interpolation,
            'Moving average (15-days window)': self.moving_average,
            'Savitzky-Golay (15-days window)': self.savitzky_golay_interpolation,
            'Harmonic': self.harmonic_function
        }

        func_name_list = list(func_dict.keys())
  

        dates_show_values = np.linspace(xtest.min(), xtest.max(), 10, dtype=np.int16)
        dates_show_string = [str(datetime(1970, 1, 1) + timedelta(days=int(d))).split()[0] for d in dates_show_values]

        # Create HTML structure
        html_content = f"""<!DOCTYPE html>
        <html>
        <head>
            <title>Results</title>
        </head>
        <body>
            <h1><u>FORCE Time-series Instant View</u></h1>
            <ul>
            <li><h2>X: <span style="color:red;">{coord_x}</span> Y: <span style="color:red;">{coord_y}</span></h2></li>
            <li><h2>Located in Tile: <span style="color:red;">{tile}</span></h2></li>
            <li><h2>From: <span style="color:red;">{start_time}</span> To: <span style="color:red;">{end_time}</span></h2></li>
            <li><h2>Sensors: <span style="color:red;">{sensor}</span></h2></li>
            <li><h2>Total clear observations: <span style="color:red;">{len(x)}</span></h2></li>
            <li><h2>Band: <span style="color:red;">{band}</span></h2></li>
            </ul>
        """

        # Generate images dynamically and store in memory
        for i in range(len(func_name_list)):
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(x, y, s=50, edgecolor='#000000', facecolor='#fcba03', label=f'Clear observations ({band})')
            try:
                if func_name_list[i] == 'Raw data':
                    pass
                elif func_name_list[i] == 'Spectral-temporal metrics':
                    ytest = func_dict[func_name_list[i]](y)
                    stm_name = ['Min', '25th', '50th', '75th', 'Max']
                    for j in range(len(ytest)):
                        ax.axhline(ytest[j], linestyle='--', color='#ff00ff')
                        ax.text(xtest.max(), ytest[j], stm_name[j], va='center', ha='left', fontsize=15, bbox=dict(facecolor='white', edgecolor='none', pad=1))
                else:
                    xtest, ytest = func_dict[func_name_list[i]](x, y, xtest)
                    ax.plot(xtest, ytest, linewidth=2, color='#ff00ff', label='Interpolation')
            except Exception as e:
                ax.text(x.max()/2, y.max()/2, 'Failed!', ha='center', va='center', fontsize=50, color='red')
            ax.set_ylim(0, y.max()*1.1)
            ax.set_xlim(xtest.min(), xtest.max())
            ax.set_xticks(dates_show_values)
            ax.set_xticklabels(dates_show_string)
            ax.set_ylabel('Reflectance (x 10,000)')
            ax.set_xlabel('Dates')
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='both', labelsize=10)
            ax.grid(visible=True, axis='both')
            ax.legend(loc="upper left")
    
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches='tight', dpi=100)
            plt.close(fig)

            # Convert image to Base64
            buf.seek(0)
            base64_str = base64.b64encode(buf.getvalue()).decode("utf-8")

            # Embed in HTML using data URI
            html_content += f'<h1>{i + 1}. {func_name_list[i]}. <button type="button">Paper</button> <button type="button">UDF code</button></h1><br>\n'
            html_content += f'<img src="data:image/png;base64,{base64_str}" alt="Image {i}"><br><br>\n'

        html_content += """</body>
        </html>
        """

        # Save the HTML file
        html_file = "FTIV_result.html"
        with open(html_file, "w") as file:
            file.write(html_content)

        print(f"HTML result created: {os.getcwd()}/{html_file}")



def print_decoded_qai(value):
    binary_str = format(value & 0xFFFF, '016b')
    binary_str = binary_str[::-1]
    decoded_string = (
        f'Valid data : {binary_str[0]} \n'
        f'Cloud state : {binary_str[1]}{binary_str[2]} \n'
        f'Cloud shadow flag : {binary_str[3]} \n'
        f'Snow flag : {binary_str[4]} \n'
        f'Water flag : {binary_str[5]} \n'
        f'Aerosol state : {binary_str[6]}{binary_str[7]} \n'
        f'Subzero flag : {binary_str[8]} \n'
        f'Saturation flag : {binary_str[9]} \n'
        f'High sun zenith flag : {binary_str[10]} \n'
        f'Illumination state : {binary_str[11]}{binary_str[12]} \n'
        f'Slope flag : {binary_str[13]} \n'
        f'Water vapor flag : {binary_str[14]} \n'
        f'Empty : {binary_str[15]} \n'
    )
    print(decoded_string)

def get_cso_value():
    filtering_list = {
        'Valid data' : ['0'],
        'Cloud state' : ['00'],
        'Cloud shadow flag' : ['0'],
        'Snow flag' : ['0'],
        'Water flag': ['0', '1'],
        'Aerosol state' : ['00'],
        'Subzero flag' : ['0'],
        'Saturation flag' : ['0'],
        'High sun zenith flag' : ['0'],
        'Illumination state' : ['00', '01', '10'],
        'Slope flag' : ['0', '1'],
        'Water vapor flag' : ['0', '1'],
        'Empty' : ['0']
    }

    cso_value = [''.join(p) for p in product(*filtering_list.values())]
    cso_value = [x[::-1] for x in cso_value]
    cso_value = [int(x, 2) for x in cso_value]
    cso_value.sort()
    return cso_value

def find_tile(yx_coords, level_2_dir, prj_file_name="datacube-definition.prj", y_first=True):
    prj_dir = os.path.join(level_2_dir, prj_file_name)

    with open(prj_dir, "r") as file:
        prj_lines = file.readlines()  

    prj_lines = [x.strip() for x in prj_lines]

    x_origin = float(prj_lines[3])
    y_origin = float(prj_lines[4])
    tile_size = float(prj_lines[5])

    coords = yx_coords.split(',')
    if y_first:
        coords = coords[::-1]
    coords = tuple([float(x) for x in coords])
    x_test = coords[0]
    y_test = coords[1]

    tile_X = int(np.floor((x_test - x_origin) / tile_size))
    tile_Y = int(np.floor((y_origin - y_test) / tile_size))

    tile_found = f"X{tile_X:04d}_Y{tile_Y:04d}"
    return tile_found, x_test, y_test


def filter_images(tile_dir, start_date, end_date, sensors='all'):
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d")

    boa_files = []

    for filename in os.listdir(tile_dir):
        if filename.endswith("BOA.tif"):
            try:
                file_date_str = filename[:8]  # Extract YYYYMMDD
                file_date = datetime.strptime(file_date_str, "%Y%m%d")
                
                if start_dt <= file_date <= end_dt:
                    boa_files.append(filename)
            except ValueError:
                continue  # Skip files that don't match the expected pattern
    
    if sensors == 'all':
        pass
    else:
        sensors_list = sensors.split(',')
        boa_files = [image for image in boa_files if any(sensor in image for sensor in sensors_list)]
    boa_files.sort()
    qai_files = [image.replace('BOA', 'QAI') for image in boa_files]

    return boa_files, qai_files

def days_since_epoch(date_str):
    # Convert string to datetime object
    date_obj = datetime.strptime(date_str, "%Y%m%d")
    
    # Define the Unix epoch start date
    epoch = datetime(1970, 1, 1)
    
    # Compute the difference in days
    return (date_obj - epoch).days

def get_band_list(band_name, boa_files):

    allowed_bands_LND = {
        'RED' : 1,
        'GREEN' : 2, 
        'BLUE': 3,
        'NIR': 4,
        'SWIR1': 5,
        'SWIR2': 6
    }

    allowed_bands_SEN = {
        'RED' : 1,
        'GREEN' : 2, 
        'BLUE': 3,
        'NIR': 8,
        'SWIR1': 9,
        'SWIR2': 10
    }

    sensor_list = [image.split('_')[2] for image in boa_files]

    band_list = []

    for sensor in sensor_list:
        if sensor[:3] == 'LND':
            band_list.append(allowed_bands_LND[band_name])
        else:
            band_list.append(allowed_bands_SEN[band_name])
    return band_list



def batch_sample_BOA(image_paths, band_list, x, y, desc='Process 1'):
    sampled_values = []
    for i in tqdm(range(len(image_paths)), desc=desc):
        with rasterio.open(image_paths[i]) as src:
            sample_generator = src.sample([(x, y)], indexes=band_list[i])
            value = next(sample_generator)[0]  # Extract the first (and only) value
        sampled_values.append(value)
    return np.array(sampled_values)

def batch_sample_QAI(image_paths, x, y, desc='Process 1'):
    sampled_values = []
    for i in tqdm(range(len(image_paths)), desc=desc):
        with rasterio.open(image_paths[i]) as src:
            sample_generator = src.sample([(x, y)], indexes=1)
            value = next(sample_generator)[0]  # Extract the first (and only) value
        sampled_values.append(value)
    return np.array(sampled_values)



def main():
    check = argparseCondition()

    parser = argparse.ArgumentParser(prog='FTIV', description="This tool produces instal overview of time-series data and interpolation methods from FORCE datacube given location X,Y", add_help=True)

    parser.add_argument(
        '-d', '--daterange',
        help='Start date and end date = date range to be considered. Valid values: YYYYMMDD,YYYYMMDD',
        default=f"20150101,{datetime.now().strftime('%Y%m%d')}",
        metavar='',
        type=check.dateRangeInput
    )
    parser.add_argument(
        '-s', '--sensors',
        help='List of sensors separated by ",". Valid values: LND05,LND08,SEN2A,... or all (Default is all)',
        default=f"all",
        type=str,
        metavar='',
    )
    parser.add_argument(
        '-b', '--band',
        help='Which band to view. Valid value: only one from [RED, GREEN, BLUE, NIR, SWIR1, SWIR2]. Default: NIR',
        choices=['RED', 'GREEN', 'BLUE', 'NIR', 'SWIR1', 'SWIR2'],
        default="NIR",
        type=str,
        metavar='',
    )
    parser.add_argument(
        '--yx',
        help='Call this if you add coordinate is in format Y,X instead of X,Y',
        action="store_true"
    )
    parser.add_argument(
        '--printarray',
        help='Call this if you only want to print out array lists of spectral values and dates. Creating report is disabled',
        action="store_true"
    )
    parser.add_argument(
        'level2Dir',
        help='FORCE datacube Level-2 directory path, the "datacube-definition.prj" file MUST exist in this directory'
    )

    parser.add_argument(
        'coordinates',
        help='Projected X,Y coordinates separated by "," (must be the same as datacube images)'
            'If you provide Y,X format, enable -yx argument'
    )

    args = parser.parse_args()


    coords = args.coordinates
    y_first = args.yx

    start_date, end_date = args.daterange

    sensors = args.sensors

    level2_dir = args.level2Dir

    band = args.band

    isprint = args.printarray

    tile, coord_x, coord_y = find_tile(coords, level2_dir, y_first=y_first)

    tile_path = os.path.join(level2_dir, tile)

    boa_files, qai_file = filter_images(tile_path, start_date=start_date, end_date=end_date, sensors=sensors)

    band_list = get_band_list(band, boa_files)

    date_list = [image[:8] for image in boa_files]
    date_list = np.array([days_since_epoch(d) for d in date_list])

    boa_files_path = [os.path.join(tile_path, image) for image in boa_files]
    qai_files_path = [os.path.join(tile_path, image) for image in qai_file]
    
    boa_values = batch_sample_BOA(boa_files_path, band_list, coord_x, coord_y, desc='Screening BOA')
    qai_values = batch_sample_QAI(qai_files_path, coord_x, coord_y, desc='Screening QAI')

    cso_value = get_cso_value()
    mask = np.isin(qai_values, cso_value)

    y_value = boa_values[mask]
    x_value = date_list[mask]

    if len(x_value) < 1:
        print('No clear observations found in the time range!')
        return 0

    if isprint:
        print('--Dates array--\n', x_value.tolist(), flush=True)
        print('--Spectral array--\n', y_value.tolist(), flush=True)
    else:
        xtest = np.linspace(days_since_epoch(start_date), days_since_epoch(end_date), 200)
        
        builder = FigureBuilder()

        builder.generate_html(x_value, y_value, xtest,
                        coord_x=coord_x,
                        coord_y=coord_y,
                        tile=tile,
                        start_time=start_date,
                        end_time=end_date,
                        sensor=sensors,
                        band=band
                        )
    return 1
    

if __name__ == "__main__":
    main()



    