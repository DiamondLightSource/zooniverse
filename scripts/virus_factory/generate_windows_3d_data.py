#!/usr/bin/env python

import argparse
import logging
import os
import os.path as op
import warnings

import h5py as h5
import numpy as np
import pandas as pd
from skimage.io import imsave
from skimage.util import img_as_ubyte

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

LOGGING_FMT = "%(asctime)s - %(levelname)s - %(message)s"
LOGGING_DATE_FMT = "%d-%b-%y %H:%M:%S"

def init_argparse() -> argparse.ArgumentParser:
    """Custom argument parser for this program.
    Returns:
        argparse.ArgumentParser: An argument parser with the appropriate
        command line args contained within.
    """
    parser = argparse.ArgumentParser(
        usage="%(prog)s [path/to/3d/image/data/file.h5] [path/to/output/directory] options...",
        description="Generates region images from a volume and fills the appropriate CSV file with metadata."
    )
    parser.add_argument(
        "-v", "--version", action="version",
        version=f"{parser.prog} version 1.0.0"
    )
    parser.add_argument("input_file_path", metavar='Input file path', type=str,
                        help='the path to a file containing 3d image data.')
    parser.add_argument("output_dir", metavar='Output directory', type=str,
                        help='the path to a directory for output.')
    parser.add_argument("--pad", metavar='Pad data flag', type=bool,
                        nargs="?", default=True,
                        help='Whether to pad the data with a median value.')
    parser.add_argument("--window", metavar='Window size', type=str,
                        nargs="?", default="240x264",
                        help='Size of the windows to extract (X,Y in pixels). e.g. 240x264')
    parser.add_argument("--overlap", metavar='Overlap size', type=str,
                        nargs="?", default="120x132",
                        help='Size of the overlap between windows (X,Y in pixels). e.g. 120x132')
    parser.add_argument("--zrange", metavar='z range', type=str,
                        nargs="?", default="start-end",
                        help='Range of the central slices to process (#start-#end). e.g. 100-400')
    parser.add_argument("--zskip", metavar='z skip', type=int,
                        nargs="?", default=1,
                        help='Number of Z slices to skip.')
    parser.add_argument("--prefix", metavar='Data prefix', type=str,
                        nargs="?", default="data",
                        help='Name for the dataset and prefix for the naming of images.')
    parser.add_argument("--fillcsv", metavar='CSV flag', type=bool,
                        nargs="?", default=True,
                        help='Whether to create a CSV file.')
    parser.add_argument("--csv_filename", metavar='CSV filename', type=str,
                        nargs="?", default="manifest.csv",
                        help='Filename for the CSV file output.')
    parser.add_argument("--transpose", metavar='Transpose data', type=str,
                        nargs="?", default=None,
                        help='Order of the axes for data transpose, if required. (e.g. 1,0,2)')
    return parser

def main():
    logging.basicConfig(
        level=logging.INFO, format=LOGGING_FMT,
        datefmt=LOGGING_DATE_FMT)
    parser = init_argparse()
    args = vars(parser.parse_args())
    logging.info('Loading data')
    source = args['input_file_path']
    # load all the data into a numpy array in memory
    with h5.File(source, "r") as f:
        data = f["/data"][()]

    # Transpose if required
    transpose = args['transpose']
    if transpose is not None:
        logging.info('Transposing the data')
        axes = tuple(map(int, transpose.split(',')))
        data = np.transpose(data, axes)

    logging.info(f"Data shape: {data.shape}")
    output = args['output_dir']
    logging.info(f"Creating output directory at {output}.")
    os.makedirs(output, exist_ok=True)

    window = args['window']
    overlap = args['overlap']
    zrange = args['zrange']
    window = tuple(map(int, window.split('x')))
    overlap = tuple(map(int, overlap.split('x')))
    zS, zE = zrange.split('-')
    zS = 0 if zS == 'start' else int(zS)
    zE = data.shape[0] if zE == 'end' else int(zE)
    
    # Pad the data
    pad_flag = args['pad']
    if pad_flag:
        # Pad the data
        logging.info("Padding the data")
        median = np.median(data[0, :, :])
        slices = data.shape[0]
        height = data.shape[1] + (2 * overlap[1])
        width = data.shape[2] + (2 * overlap[0])
        padded = np.zeros((slices, height, width))
        for i in range(slices):
            padded[i, :, :] = (np.pad(data[i, :, :],
            ((overlap[1], overlap[1]), (overlap[0], overlap[0])), 'constant', constant_values=median))    

        data = padded
        logging.info(f"Padded Data shape: {data.shape}")
    
    zskip = args['zskip']
    fillcsv = args['fillcsv']
    prefix = args['prefix']
    csv_list = []
    for z in range(zS, zE, zskip):
        logging.info(f"Processing slice: {z}")
        for y in range(0, data.shape[1], window[1] - overlap[1]):
            yS = min(y, data.shape[1] - window[1]); yE = yS + window[1]
            for x in range(0, data.shape[2], window[0] - overlap[0]):
                xS = min(x, data.shape[2] - window[0]); xE = xS + window[0]
                roi = data[z, yS:yE, xS:xE]
                fname = '{}_{}_{}x{}x{}x{}.png'.format(prefix, z, xS, xE, yS, yE)
                if np.issubdtype(roi.dtype, np.float):
                    roi = img_as_ubyte(roi)
                imsave(op.join(output, fname), roi)
                if fillcsv:
                    csv_list.append([fname, prefix, z, xS, xE, yS, yE, source, transpose])

    csvfile = args['csv_filename']
    if fillcsv:
        columns = ['filename', 'prefix', 'slice', 'xstart', 'xend', 'ystart', 'yend', '#source', '#transpose']
        df = pd.DataFrame(csv_list, columns=columns)
        df.index.name = 'id'
        csvfile = op.join(output, csvfile)
        if op.isfile(csvfile):
            with open(csvfile, 'a') as f:
                df.to_csv(f, header=False)
        else:
            df.to_csv(csvfile)

if __name__ == "__main__":
    main()
