# Module for loading, processing, and visualizing LRO data (level 2 data from CRaTER in particular).

#### Top-level imports ####
import os, pickle, csv
import numpy as np
from datetime import datetime, timedelta
import pooch, time
from tqdm import tqdm
import gzip
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
from scipy.interpolate import InterpolatedUnivariateSpline
import pandas as pd
from lunarsky import MoonLocation
import astropy.units as u
from scipy.interpolate import griddata
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable

#### Local imports ####
from srcDev.tools import toolbox

#### Global Variables ####
download_path = 'CRaTER'
process_path = 'CRaTER_processed'
model_path = 'modelData'

#### Helper Functions ####
def getCRaTER(dateStart, dateEnd):
    """
    Downloads Level 2 CRaTER data to a default location. Does so between two dates inclusive. To download only for a
    single day, just set the ending date to equal the starting date.
    :param dateStart: str
        Starting date for data to download in YYYY-MM-DD format.
    :param dateEnd:
        Ending date for data to download in YYYY-MM-DD format.
    :return: None
    """
    # Convert the starting and ending dates to datetimes:
    dateStartDatetime = datetime.strptime(dateStart, "%Y-%M-%d")
    dateEndDatetime = datetime.strptime(dateEnd, "%Y-%M-%d")
    # Make an array of datetimes between the starting and ending dates:
    dateTimeArray = toolbox.linspace_datetime(dateStart, dateEnd)
    # Make an array of strings for the datetimes between the starting and ending dates:
    dateArray = np.array([str(element)[:10] for element in dateTimeArray])
    # Do the same as the above but with the format of YYYYDOY:
    dateArray_mod = np.array([x[:4]+toolbox.dayNumber(y) for x,y in zip(dateArray,dateTimeArray)])

    # Set up the URLs and download the Level 2 LRO-CRaTER data (use Pooch to do so, to avoid unnecessary overwriting):
    top_url = 'https://crater-products.sr.unh.edu/data/inst/l2/'
    url_list = [top_url+element+'/' for element in dateArray_mod]
    for i in tqdm(range(len(url_list))):
        # Download the zipped PRIMARY SCIENCE Level 2 data:
        current_CRaTER_file = 'CRAT_L2_PRI_'+dateArray_mod[i]+'_V01.TAB.gz'
        pooch.retrieve(url=url_list[i]+current_CRaTER_file, known_hash=None, fname=current_CRaTER_file, path=download_path)
        # Download the zipped SECONDARY SCIENCE Level 2 data:
        current_CRaTER_file = 'CRAT_L2_SEC_' + dateArray_mod[i] + '_V01.TAB.gz'
        pooch.retrieve(url=url_list[i] + current_CRaTER_file, known_hash=None, fname=current_CRaTER_file,
                       path=download_path)
        # TODO: RETRIEVE GEOMETRY FILES (LRO ephemeris data): https://crater-products.sr.unh.edu/data/sc/asflown/ephem_pntg/

    # Print, exit and return nothing:
    time.sleep(0.5)
    print('LRO files downloaded/obtained.')
    time.sleep(0.5)
    return

def processCRaTER(filename, des='primary', override=False):
    """
    Given a Level 2 CRaTER file, read it and return its contents as a dictionary. The data is saved to a default
    location, and if it already exists, it is simple read in.
    Reference - Tables 33 and 34 here: https://snebulos.mit.edu/projects/crater/file_cabinet/0/01211/01211_rC.pdf
    :param filename: str
        Location of the file to read.
    :param des: str
        Either 'primary' or 'secondary'. Denotes whether the data is primary or secondary data. Default is 'primary'.
    :param override: bool
        Determines whether existing data (if present) is re-downloaded and overloaded.
    :return dataDict: dictionary
        A dictionary with keys containing the data in each field, with fields corresponding to either Primary science
        data or Secondary science data.
    """
    # Unzip (open) the file (TODO: Make it so that this data is not re-unzipped if that is already done):
    with gzip.open(download_path+'/'+filename, 'r') as f:
        file_content = f.readlines()

    # Read in the data: TODO: ADD A CONDITIONAL FOR READING GEOMETRY FILES
    if des=='primary':
        dictFile = process_path+'/'+filename[:-7] + '_primary.pkl'
        if os.path.isfile(dictFile) == False or override == True:
            timeVals = [] # UTC
            amplitude = [] # Amplitude in Detectors D1 to D6 (eV)
            energy = [] # Energy deposited in Detectors D1 to D6 (eV)
            lineal_energy_transfer = [] # Lineal Energy Transfer in silicon in Detectors D1 to D6 (eV/microns)
            for line in file_content:
                my_line = line.split(b',')
                #
                timeVals.append( datetime.strptime(my_line[2].decode('utf-8'),'"%Y-%m-%dT%H:%M:%S"') + timedelta(milliseconds=float(my_line[3])) )
                #
                ampList = [float(element.decode('utf-8')) for element in my_line[10:16]]
                amplitude.append(ampList)
                #
                energyList = [float(element.decode('utf-8')) for element in my_line[16:22]]
                energy.append(energyList)
                #
                letList = [float(element.decode('utf-8')) for element in my_line[22:28]]
                lineal_energy_transfer.append(letList)
            dataDict = {
                'Time': timeVals,
                'Amplitude': amplitude,
                'Energy': energy,
                'LET': lineal_energy_transfer
            }
            with open(dictFile, 'wb') as handle:
                pickle.dump(dataDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(dictFile, 'rb') as handle:
                dataDict = pickle.load(handle)
            print('Loaded existing processed data.')
    else:
        dictFile = process_path+'/'+filename[:-7] + '_secondary.pkl'
        if os.path.isfile(dictFile) == False or override == True:
            timeVals = [] # UTC
            altitude = [] # Height above the lunar surface; km
            latitude = [] # Selenocentric S/C latitude (1AU); degrees
            longitude = [] # Selenocentric S/C longitude (1AU); degrees
            for line in file_content:
                my_line = line.split(b',')
                #
                timeVals.append(datetime.strptime(my_line[2].decode('utf-8'), '"%Y-%m-%dT%H:%M:%S"'))
                # Selenodetic Coordinates:
                coords = MoonLocation.from_selenocentric(float(my_line[-3].decode('utf-8')), float(my_line[-2].decode('utf-8')), float(my_line[-1].decode('utf-8')), unit=u.km)
                coords_sd = coords.to_selenodetic()
                latitude.append(coords_sd.lon.value)
                longitude.append(coords_sd.lat.value)
                altitude.append(coords_sd.height.value)
            dataDict = {
                'Time': timeVals,
                'Altitude': altitude,
                'Latitude': latitude,
                'Longitude': longitude
            }
            with open(dictFile, 'wb') as handle:
                pickle.dump(dataDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(dictFile, 'rb') as handle:
                dataDict = pickle.load(handle)
            print('Loaded existing processed data.')
    # Return the data:
    return dataDict

def displayCRaTER(dateStart, dateEnd):
    """
    Downloads / read in CRaTER data between two dates and displays it on in selenocentric coordinates.
    :param dateStart: str
        Starting date for data to download in YYYY-MM-DD format.
    :param dateEnd:
        Ending date for data to download in YYYY-MM-DD format.
    :return:
    """
    # Convert the starting and ending dates to datetimes:
    dateStartDatetime = datetime.strptime(dateStart, "%Y-%M-%d")
    dateEndDatetime = datetime.strptime(dateEnd, "%Y-%M-%d")
    # Make an array of datetimes between the starting and ending dates:
    dateTimeArray = toolbox.linspace_datetime(dateStart, dateEnd)
    # Make an array of strings for the datetimes between the starting and ending dates:
    dateArray = np.array([str(element)[:10] for element in dateTimeArray])
    # Do the same as the above but with the format of YYYYDOY:
    dateArray_mod = np.array([x[:4] + toolbox.dayNumber(y) for x, y in zip(dateArray, dateTimeArray)])

    # Get a list of the filenames to be read in...
    # Primary Data:
    primaryList = ['CRAT_L2_PRI_'+x+'_V01.TAB.gz' for x in dateArray_mod]
    # Secondary Data:
    secondaryList = ['CRAT_L2_SEC_'+x+'_V01.TAB.gz' for x in dateArray_mod]

    # Read in data from each day into a .csv (writing line by line):
    print('Aggregating CRaTER data for display...')
    time.sleep(1)
    # aggregatedDict = {
    #     'All_Detection_Times': np.array([]),
    #     'All_Detection_Alts': np.array([]),
    #     'All_Detection_Lats': np.array([]),
    #     'All_Detection_Lons': np.array([])
    # }
    csv_name = model_path+'/modelData_'+dateStart+'-'+dateEnd+'.csv'
    # If the .csv file already exists, just open it and append to it. Otherwise, make a new one and write to it:
    if os.path.isfile(csv_name) == True:
        with open(csv_name, 'a') as modelFile:
            writer = csv.writer(modelFile, delimiter=' ') # " ".join(map(str, names))+'\n')
            for i in tqdm(range(len(dateArray))):
                print('Day ' + str(i + 1) + ': ' + dateArray[i] + '...')
                time.sleep(0.25)
                # Grab primary files:
                primaryDict = processCRaTER(primaryList[i], des='primary')  # , override=True)
                # Grab secondary files:
                secondaryDict = processCRaTER(secondaryList[i], des='secondary')  # , override=True)

                # Interpolate the times and locations in the secondary files to the same cadence as the primary files; geolocate the measurements in the primary files:
                fname = model_path + '/' + dateArray_mod[i] + '_harmonized.pkl'
                harmonizedDict, pre_existing_data = harmonizeCrater(primaryDict, secondaryDict,
                                                                    fname)  # , override=True)

                # Write the new data to the .csv file:
                if pre_existing_data == False:
                    for j in range(len(harmonizedDict['Detection_Times'])):
                        row_text = [str(harmonizedDict['Detection_Times'][j]), str(harmonizedDict['Detection_Alts'][j]),
                                    str(harmonizedDict['Detection_Lats'][j]), str(harmonizedDict['Detection_Lons'][j]),
                                    str(harmonizedDict['D1_Meas'][j]), str(harmonizedDict['D2_Meas'][j])]
                        writer.writerow(row_text)
                else:
                    print('This day has already been written to the .csv file; skipping...')
    else:
        with open(csv_name, 'w') as modelFile:
            writer = csv.writer(modelFile, delimiter=' ')
            names = ['All_Detection_Times', 'All_Detection_Alts', 'All_Detection_Lats', 'All_Detection_Lons', 'D1_Meas', 'D2_Meas']
            writer.writerow(names) #" ".join(map(str, names))+'\n')
            for i in tqdm(range(len(dateArray))):
                print('Day '+str(i+1)+': '+dateArray[i]+'...')
                time.sleep(0.25)
                # Grab primary files:
                primaryDict = processCRaTER(primaryList[i], des='primary') #, override=True)
                # Grab secondary files:
                secondaryDict = processCRaTER(secondaryList[i], des='secondary') #, override=True)

                # Interpolate the times and locations in the secondary files to the same cadence as the primary files; geolocate the measurements in the primary files:
                fname = model_path+'/'+dateArray_mod[i]+'_harmonized.pkl'
                harmonizedDict, pre_existing_data = harmonizeCrater(primaryDict, secondaryDict, fname)#, override=True)

                # Append the proton GCR flux into an ever-growing data structure:
                # aggregatedDict.update({'All_Detection_Times': np.concatenate(
                #     (aggregatedDict['All_Detection_Times'], harmonizedDict['Detection_Times']))})
                # aggregatedDict.update({'All_Detection_Times': np.concatenate(
                #     (aggregatedDict['All_Detection_Alts'], harmonizedDict['Detection_Alts']))})
                # aggregatedDict.update({'All_Detection_Lats': np.concatenate(
                #     (aggregatedDict['All_Detection_Lats'], harmonizedDict['Detection_Lats']))})
                # aggregatedDict.update({'All_Detection_Lons': np.concatenate(
                #     (aggregatedDict['All_Detection_Lons'], harmonizedDict['Detection_Lons']))})

                # Write the new data to the .csv file:
                for j in range(len(harmonizedDict['Detection_Times'])):
                    row_text = [str(harmonizedDict['Detection_Times'][j]).replace(' ', 'T'), str(harmonizedDict['Detection_Alts'][j]),
                                str(harmonizedDict['Detection_Lats'][j]), str(harmonizedDict['Detection_Lons'][j]),
                                str(harmonizedDict['D1_Meas'][j]), str(harmonizedDict['D2_Meas'][j])]
                    writer.writerow(row_text)

    # TODO: Visualize the aggregated data in Lunar Geographic Coordinates:
    # (1): Read in the geolocated detections in the .csv file:
    df = pd.read_csv(csv_name, sep=' ')

    # (2): Bin detections into the grid (default is 15 degree resolution):
    step = 15
    to_bin = lambda x: np.floor(x / step) * step
    df_detections = df[['All_Detection_Lats', 'All_Detection_Lons', 'D1_Meas', 'D2_Meas']]
    df_detections['dose'] = df_detections['D1_Meas'] + df_detections['D2_Meas']
    df_detections['latBin'] = to_bin(df_detections['All_Detection_Lats'])
    df_detections['lonBin'] = to_bin(df_detections['All_Detection_Lons'])
    # df_dose = df_detections[['dose', 'latBin', 'lonBin']]
    # lunarMap = df_dose.groupby(['lonBin', 'latBin']).size().unstack().fillna(0)

    # (3): Display and save the resulting image:
    points = list(zip(df['All_Detection_Lons'], df['All_Detection_Lats']))
    values = df_detections['dose'].values * 1e-19 * 100
    rRes = 15
    xRange = np.arange(-180, 180+rRes, rRes)
    yRange = np.arange(-90, 90+rRes, rRes)
    gridX, gridY = np.meshgrid(xRange, yRange)
    gridPh = griddata(points, values, (gridX, gridY), method='cubic')
    gridPhf = np.nan_to_num(gridPh, copy=True, nan=0.0)
    lunarBkg = np.asarray(Image.open('lroc_color_poles_1k.jpg'))
    plt.figure()
    ax = plt.gca()
    im1 = ax.imshow(lunarBkg, extent=[-180, 180, 90, -90])
    im2 = ax.imshow(gridPhf, cmap='viridis', interpolation='bicubic', extent=[-180, 180, 90, -90], alpha=0.8)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im2, cax=cax)
    cbar.set_label('Dose (cG)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Lunar GCR Dose: '+dateStart+' to '+dateEnd)
    plt.savefig('figures/Lunar_GCR_Map_'+dateStart+'_'+dateEnd+'.png', dpi=300)

    print('Model data available here: '+model_path+'/modelData_'+dateStart+'-'+dateEnd+'.csv')

def harmonizeCrater(primDict, secondDict, fname, override=False):
    """
    Given a dict of LRO primary and secondary data, generate a NEW dictionary of geolocated GCR measurements.
    :param primDict: dictionary
        Dictionary object with keys: 'Time', 'Amplitude', 'Energy', 'LET'
    :param secondDict: dictionary
        Dictionary object with keys: 'Time', 'Altitude', 'Latitude', 'Longitude'
    :param fname: str
        String with the place to download the harmonized data. REQUIRED.
    :param override: bool
        Determines whether pre-existing data is downloaded again.
    :return harmonizedDict: dictionary
        Dictionary with keys: 'Time', 'Altitude', 'Latitude', 'Longitude', 'Amplitude', 'Energy', 'LET'
    :return pre_existing_data: bool
        Indicates whether data had to be processed or whether it was just read in.
    """
    # TODO: Modify this function to use location data from the geometry files!
    if os.path.isfile(fname) == False or override == True:
        # Loop through each day and interpolate in location per day:
        secondDict_times_seconds = np.array([((element - secondDict['Time'][0]).seconds) + ((element - secondDict['Time'][0]).microseconds)*(1e-6) for element in secondDict['Time']])
        interp_times_seconds = np.linspace(secondDict_times_seconds[0], secondDict_times_seconds[-1], num=len(primDict['Time']))
        #
        interp_alt_spl = InterpolatedUnivariateSpline(secondDict_times_seconds, secondDict['Altitude'], k=3)
        interp_lat_spl = InterpolatedUnivariateSpline(secondDict_times_seconds, secondDict['Latitude'], k=3)
        interp_lon_spl = InterpolatedUnivariateSpline(secondDict_times_seconds, secondDict['Longitude'], k=3)
        #
        interp_alts = interp_alt_spl(interp_times_seconds)
        interp_lats = interp_lat_spl(interp_times_seconds)
        interp_lons = interp_lon_spl(interp_times_seconds)
        # Extract the GCR detections only to calculate dose rate per day (cGy/day) - uses technique of Schawdron, et al. 2012 (doi:10.1029/2011JE003978)
        d1_measurements = np.asarray([element[0] for element in primDict['Energy']])
        d2_measurements = np.asarray([element[1] for element in primDict['Energy']])
        good_adu_d1_inds = np.where((d1_measurements >= 38) & (d1_measurements <= 4096))[0]
        good_adu_d2_inds = np.where((d2_measurements >= 7) & (d2_measurements <= 920))[0]
        d1_dose = sum(d1_measurements[good_adu_d1_inds])
        d2_dose = sum(d2_measurements[good_adu_d2_inds])
        dos_cGy_day = (d1_dose + d2_dose) * 1e-19 * 100
        num_detections = len(good_adu_d1_inds) + len(good_adu_d2_inds)
        # GCR Detections (geolocated)
        all_good_adu_inds = np.concatenate((good_adu_d1_inds, good_adu_d2_inds))
        all_good_adu_inds_unique = np.unique(all_good_adu_inds)
        detection_times = np.asarray(primDict['Time'])[all_good_adu_inds_unique]
        detection_alts = interp_alts[all_good_adu_inds_unique]
        detection_lats = interp_lats[all_good_adu_inds_unique]
        detection_lons = interp_lons[all_good_adu_inds_unique]
        # It is strongly recommended to limit the amount of processed data that is saved. This makes I/O operations faster:
        harmonizedDict = {
            'Start_Time': secondDict['Time'][0],
            'End_Time': secondDict['Time'][-1],
            # 'Time': interp_times_seconds,
            # 'Altitude': interp_alts,
            # 'Latitude': interp_lats,
            # 'Longitude': interp_lons,
            # 'Amplitude': primDict['Amplitude'],
            # 'Energy': primDict['Energy'],
            # 'LET': primDict['LET'],
            'Total_Dose': dos_cGy_day,
            'Num_GCR_Detections': num_detections,
            # 'd1_Detections': d1_measurements[good_adu_d1_inds],
            # 'd2_Detections': d2_measurements[good_adu_d2_inds],
            'Detection_Times': detection_times,
            'Detection_Alts': detection_alts,
            'Detection_Lats': detection_lats,
            'Detection_Lons': detection_lons,
            'D1_Meas': d1_measurements[all_good_adu_inds_unique],
            'D2_Meas': d2_measurements[all_good_adu_inds_unique]
        }
        with open(fname, 'wb') as handle:
            pickle.dump(harmonizedDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Data obtained.')
        pre_existing_data = False
    else:
        with open(fname, 'rb') as handle:
            harmonizedDict = pickle.load(handle)
        print('Loaded existing processed data.')
        pre_existing_data = True

    return harmonizedDict, pre_existing_data

def fitCRaTER(dataFile):
    """
    Given a large .csv file of CRaTER data, fit a spherical harmonic model expansion to the data, with techniques
    similar to that Yu, et al. 2022 (https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1029/2022sw003113)
    or Zhang and Zhao 2019 (https://doi.org/10.1016/j.asr.2018.10.031).
    :param dataFile:
    :return modelCoefs: numpy.ndarray
        Coefficients of the spherical harmonic model.
    """
    return

#### Execution (examples) ####
if __name__ == '__main__':
    dateStart = '2012-05-01'
    dateEnd = '2012-05-05'

    # Download CRaTER Level 2 data between two dates:
    getCRaTER(dateStart, dateEnd)

    # Open up a single CRaTER file (primary science data):
    # filename1 = 'CRAT_L2_PRI_2012122_V01.TAB.gz'
    # dict1 = processCRaTER(filename1, des='primary')

    # Open up a single CRaTER file (secondary science data):
    # filename2 = 'CRAT_L2_SEC_2012122_V01.TAB.gz'
    # dict2 = processCRaTER(filename2, des='secondary')

    # Plotting CRaTER Galactic Cosmic Ray data:
    displayCRaTER(dateStart, dateEnd)

    ellipsis

