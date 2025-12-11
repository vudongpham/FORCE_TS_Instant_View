import rasterio
from pyproj import Transformer
import numpy as np
from itertools import product
import os
from datetime import datetime, timedelta
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import io
import base64
import os
import argparse
import re
import warnings
from ftiv import ftiv_version
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
    def __init__(self, debug=False):
        self.debug = debug
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
    
    def rbf_ensemble_interpolation(self, x, y, xtest, rbf_sigma=[30, 60, 90]):
        def compute_rbf_weights(time_series, center, fwhm):
            """Computes RBF weights using a Gaussian kernel."""
            sigma = fwhm / 2.355
            return np.exp(-((time_series - center) ** 2) / (2 * sigma ** 2))

        def apply_rbf_interpolation(x_train, y_train, x_test, fwhm, user_weight, cutoff_value):
            """Applies RBF interpolation to predict values at x_test points."""
            predictions = []
            weights_list = []
            for test_point in x_test:
                weights = compute_rbf_weights(x_train, test_point, fwhm)
                valid_mask = weights >= cutoff_value
                if not np.any(valid_mask):  # Avoid division by zero or empty data
                    predictions.append(np.nan)
                    weights_list.append(0)
                    continue
                # Apply weights only to valid data points
                y_valid = y_train[valid_mask]
                weights_valid = weights[valid_mask]

                weighted_sum = np.sum(y_valid * weights_valid)
                weight_sum = np.sum(weights_valid)

                interpolated_value = weighted_sum / weight_sum if weight_sum != 0 else np.nan
                availability_score = (weight_sum / np.sum(weights)) * user_weight

                predictions.append(interpolated_value)
                weights_list.append(availability_score)

            return np.array(predictions), np.array(weights_list)

        def apply_ensemble_rbf_interpolation(x_train, y_train, x_test, fwhm_list, user_weights, cutoff_value=0.01):
            """Applies ensemble RBF interpolation using multiple kernel sizes."""
            if user_weights is None:
                user_weights = [1.0] * len(fwhm_list)
            assert len(fwhm_list) == len(user_weights), "Mismatch between FWHM list and user weights"

            all_predictions, all_weights = [], []
            for fwhm, user_weight in zip(fwhm_list, user_weights):
                pred, weight = apply_rbf_interpolation(x_train, y_train, x_test, fwhm, user_weight, cutoff_value)
                all_predictions.append(pred)
                all_weights.append(weight)

            # Weighted ensemble averaging
            sum_weights = np.sum(all_weights, axis=0)
            ensemble_prediction = np.sum(np.multiply(all_predictions, all_weights), axis=0) / sum_weights

            return ensemble_prediction

        def convert_days_to_decimal_year(days_since_epoch):
            """Converts days since epoch (1970-01-01) to decimal year format."""
            date = datetime(1970, 1, 1) + timedelta(days=int(days_since_epoch))
            year = date.year
            day_of_year = date.timetuple().tm_yday
            total_days = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
            return year + (day_of_year - 1) / total_days

        # Convert input timestamps to decimal years
        x_unique, indices = np.unique(x, return_index=True)
        y_unique = y[indices]
        x_unique = np.array([convert_days_to_decimal_year(d) for d in x_unique])
        x_test_new = np.array([convert_days_to_decimal_year(d) for d in xtest])

        # RBF settings
        rbf_cutoff = 0.01  # Minimum weight threshold
        rbf_fwhms = [d / 365 for d in rbf_sigma]  # RBF kernel sizes in decimal years
        rbf_user_weights = [1, 1, 1]  # Relative weights of different scales

        # Perform ensemble RBF interpolation
        y_test = apply_ensemble_rbf_interpolation(x_unique, y_unique, x_test_new, rbf_fwhms, rbf_user_weights, rbf_cutoff)
        
        return xtest, y_test
    
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
        xtest_feed = np.concatenate([xtest - 365, xtest, xtest + 365])
        ytest = func_(xtest_feed, *popt)
        ytest = ytest[len(xtest): len(xtest) * 2]
        return xtest, ytest
    
    def savitzky_golay_interpolation(self, x, y, xtest,window_size=30, polyorder = 2):
        ytest = np.interp(xtest, x, y)
        ytest = savgol_filter(ytest, window_size, polyorder, mode='nearest')
        return xtest, ytest
    
    def generate_html(self, x, y, xtest, coord_x, coord_y, x_lat, x_lon, tile, start_time, end_time, sensor, band):
        func_dict = {
            'Raw data': 0,
            'Spectral-temporal metrics': self.spectral_temporal_metrics,
            'Linear interpolation': self.linear_interpolation,
            'Moving average': self.moving_average,
            'RBF interpolation': self.rbf_ensemble_interpolation,
            'Harmonic': self.harmonic_function,
            'Savitzky-Golay': self.savitzky_golay_interpolation
        }

        func_name_list = list(func_dict.keys())
  

        dates_show_values = np.linspace(xtest.min(), xtest.max(), 10, dtype=np.int16)
        dates_show_string = [str(datetime(1970, 1, 1) + timedelta(days=int(d))).split()[0] for d in dates_show_values]

        # Create HTML structure
        html_content = f"""<!DOCTYPE html>
        <html>
        <head>
            <title>FTIV results</title>
        </head>
        <body>
            <h1><u>FORCE Time-series Instant View</u></h1>
            <ul>
            <li><h2>Latitute: <span style="color:red;">{x_lat:.2f}</span> Longitute: <span style="color:red;">{x_lon:.2f}</span></h2></li>
            <li><h2>Datacube projection X: <span style="color:red;">{coord_x:.2f}</span> Y: <span style="color:red;">{coord_y:.2f}</span></h2></li>
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
                if self.debug:
                    print(e)
                ax.text(x.max()/2, y.max()/2, 'Failed!', ha='center', va='center', fontsize=50, color='red')
            ymin = y.min() - 1000
            ymax = y.max() + 1000
            ax.set_ylim(ymin, ymax)
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

def get_cso_value(best_quality=False):
    filtering_default = {
        'Valid data' : ['0'],
        'Cloud state' : ['00'],
        'Cloud shadow flag' : ['0'],
        'Snow flag' : ['0'],
        'Water flag': ['0', '1'],
        'Aerosol state' : ['00', '01', '10', '11'],
        'Subzero flag' : ['0'],
        'Saturation flag' : ['0'],
        'High sun zenith flag' : ['0', '1'],
        'Illumination state' : ['00', '01', '10', '11'],
        'Slope flag' : ['0', '1'],
        'Water vapor flag' : ['0', '1'],
        'Empty' : ['0']
    }

    filtering_best = {
        'Valid data' : ['0'],
        'Cloud state' : ['00'],
        'Cloud shadow flag' : ['0'],
        'Snow flag' : ['0'],
        'Water flag': ['0', '1'],
        'Aerosol state' : ['00'],
        'Subzero flag' : ['0'],
        'Saturation flag' : ['0'],
        'High sun zenith flag' : ['0'],
        'Illumination state' : ['00'],
        'Slope flag' : ['0', '1'],
        'Water vapor flag' : ['0', '1'],
        'Empty' : ['0']
    }

    if best_quality:
        filtering_list = filtering_best
    else:
        filtering_list = filtering_default
    cso_value = [''.join(p) for p in product(*filtering_list.values())]
    cso_value = [x[::-1] for x in cso_value]
    cso_value = [int(x, 2) for x in cso_value]
    cso_value.sort()
    return cso_value

def find_tile(lat, lng, level_2_dir, prj_file_name="datacube-definition.prj"):

    def extract_projection(f_string):
        if '=' in f_string:
            f_string = f_string.split('=', 1)[1]
        return f_string.lstrip().lstrip('*').strip()

    def extract_float(f_string):
        pattern = r"[+-]?(?:\d+\.\d*|\.\d+|\d+)"
        value = float(re.search(pattern, f_string).group())
        return value
    
    prj_dir = os.path.join(level_2_dir, prj_file_name)

    with open(prj_dir, "r") as file:
        prj_lines = file.readlines()  

    prj_lines = [x.strip() for x in prj_lines]

    proj_wkt = extract_projection(prj_lines[0])

    target_crs = rasterio.CRS.from_wkt(proj_wkt)

    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)

    x_origin = float(extract_float(prj_lines[3]))
    y_origin = float(extract_float(prj_lines[4]))
    tile_size_X = float(extract_float(prj_lines[5]))
    tile_size_Y = float(extract_float(prj_lines[6]))

    x_test, y_test = transformer.transform(lng, lat)

    tile_X = int(np.floor((x_test - x_origin) / tile_size_X))
    tile_Y = int(np.floor((y_origin - y_test) / tile_size_Y))

    tile_found = f"X{tile_X:04d}_Y{tile_Y:04d}"

    return tile_found, x_test, y_test, lat, lng


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
        'NDVI': 0,
        'RED' : 1,
        'GREEN' : 2, 
        'BLUE': 3,
        'NIR': 4,
        'SWIR1': 5,
        'SWIR2': 6
    }

    allowed_bands_SEN = {
        'NDVI': 0,
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

def batch_sample_BOA_NDVI(image_paths, boa_files, x, y, desc='Process 1'):
    red_list = []
    nir_list = []
    for i in tqdm(range(len(image_paths)), desc=desc):
        sensor = boa_files[i].split('_')[2]
        if sensor[:3] == 'LND':
            red_band = 3
            nir_band = 4
        else:
            red_band = 3
            nir_band = 8
        
        with rasterio.open(image_paths[i]) as src:
            generator = src.sample([(x, y)], indexes=red_band)
            red_value = next(generator)[0]
            generator = src.sample([(x, y)], indexes=nir_band)
            nir_value = next(generator)[0]
            
        red_list.append(red_value)
        nir_list.append(nir_value) 
    red_list = np.array(red_list)
    nir_list = np.array(nir_list)
    ndvi = ((nir_list - red_list) / (nir_list + red_list)) * 10000.
    ndvi = ndvi.astype(np.int16)
    return ndvi  

def batch_sample_QAI(image_paths, x, y, desc='Process 1'):
    sampled_values = []
    for i in tqdm(range(len(image_paths)), desc=desc):
        with rasterio.open(image_paths[i]) as src:
            sample_generator = src.sample([(x, y)], indexes=1)
            value = next(sample_generator)[0]  # Extract the first (and only) value
        sampled_values.append(value)
    return np.array(sampled_values)



def main():
    current_version = ftiv_version

    check = argparseCondition()

    parser = argparse.ArgumentParser(prog='FTIV', description=f"FTIV version {current_version}\n.This tool produces instal overview of time-series data and interpolation methods from FORCE datacube given location X,Y", add_help=True)

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
        help='Which band to view. Valid value: only one from [RED, GREEN, BLUE, NIR, SWIR1, SWIR2, NDVI]. Default: NDVI',
        choices=['RED', 'GREEN', 'BLUE', 'NIR', 'SWIR1', 'SWIR2', 'NDVI'],
        default="NDVI",
        type=str,
        metavar='',
    )

    parser.add_argument(
        '--printarray',
        help='Call this if you only want to print out array lists of spectral values and dates. Creating report is disabled',
        action="store_true"
    )

    parser.add_argument(
        '--bestquality',
        help='Enables cloud masking more aggressivly, resulting better quality observations, but potentially less quantity. Default: False',
        action="store_true",
    )

    parser.add_argument(
        'level2Dir',
        help='FORCE datacube Level-2 directory path, the "datacube-definition.prj" file MUST exist in this directory'
    )

    parser.add_argument(
        'coordinates',
        help='Geographic coordinates (Lat,Lon) (Y,X) separated by ","'

    )

    args = parser.parse_args()

    coords = args.coordinates

    start_date, end_date = args.daterange

    sensors = args.sensors

    level2_dir = args.level2Dir

    band = args.band

    isprint = args.printarray

    isbest_qai = args.bestquality

    tile, coord_x, coord_y, x_lat, x_lon = find_tile(coords, level2_dir)

    tile_path = os.path.join(level2_dir, tile)

    boa_files, qai_file = filter_images(tile_path, start_date=start_date, end_date=end_date, sensors=sensors)

    band_list = get_band_list(band, boa_files)

    date_list = [image[:8] for image in boa_files]
    date_list = np.array([days_since_epoch(d) for d in date_list])

    boa_files_path = [os.path.join(tile_path, image) for image in boa_files]
    qai_files_path = [os.path.join(tile_path, image) for image in qai_file]
    
    if band == 'NDVI':
        boa_values = batch_sample_BOA_NDVI(boa_files_path, boa_files, coord_x, coord_y, desc='Screening BOA')
    else:
        boa_values = batch_sample_BOA(boa_files_path, band_list, coord_x, coord_y, desc='Screening BOA')
    qai_values = batch_sample_QAI(qai_files_path, coord_x, coord_y, desc='Screening QAI')

    cso_value = get_cso_value(best_quality=isbest_qai)
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
        
        builder = FigureBuilder(debug=False)

        builder.generate_html(x_value, y_value, xtest,
                        coord_x=coord_x,
                        coord_y=coord_y,
                        x_lat=x_lat,
                        x_lon=x_lon,
                        tile=tile,
                        start_time=start_date,
                        end_time=end_date,
                        sensor=sensors,
                        band=band
                        )
    return 1
    

if __name__ == "__main__":
    main()