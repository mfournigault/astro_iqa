import os
import uuid
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
import argparse

def list_fits_files(data_path, file_type=".fit"):
    """
    List all .fit files in the specified directory.
    """
    return [f for f in os.listdir(data_path) if f.endswith(file_type)]

def generate_ldac_files(data_path, fits_files, file_type=".fit", config_file="xymfhe.sex"):
    """
    Generate LDAC files using sextractor for each .fit file.
    """
    for fits_file in fits_files:
        fits_path = os.path.join(data_path, fits_file)
        ldac_path = fits_path.replace(file_type, ".ldac")
        if not os.path.exists(ldac_path):
            os.system(f"sex {fits_path} -c {config_file}")
            os.system(f"mv test.cat {ldac_path}")

def log_scale_data(data, mini, maxi):
    """
    Apply logarithmic scaling to the data within the specified range.
    """
    data = np.where(data < mini, mini, data)
    data = np.where(data > maxi, maxi, data)
    data = np.log10(data)
    return data

def remove_outliers(data, mini, maxi):
    """
    Remove outliers from the data by clipping values outside the specified range.
    """
    data = np.where(data < mini, mini, data)
    data = np.where(data > maxi, maxi, data)
    return data

def process_ldac_files(data_path, catalog,file_type=".fit"):
    """
    Process LDAC files and update the catalog with the extracted data.
    """
    for ldac_file in [f for f in os.listdir(data_path) if f.endswith(".ldac")]:
        fits_id, extension = os.path.splitext(ldac_file)
        ldac_path = os.path.join(data_path, ldac_file)
        print(f"Processing {ldac_path}")
        ldac = fits.open(ldac_path)
        # print(ldac.info())
        ldac_tables = [hdu for hdu in ldac if isinstance(hdu, fits.BinTableHDU)]
        n_ccd = len(ldac_tables)
        fits_ = fits.open(os.path.join(data_path, fits_id + file_type))
        # print(fits_.info())
        for i in range(1, len(ldac)):
            print(f"Processing CCD {i}")
            if ldac[i].data is not None and ldac[i].data.shape[0] > 0:
                table = Table(ldac[i].data)
                table.add_column(fits_id, name="FITS_ID", index=0)
                table.add_column(np.uint8(i), name="CCD_ID", index=1)
                ids = [str(uuid.uuid4().hex) for _ in range(ldac[i].data.shape[0])]
                table.add_column(ids, name="OBJECT_ID", index=2)
                try:
                    if file_type == ".fits" or file_type == ".fits.gz":
                        exptime = fits_[i].header["EXPTIME"]
                    else:
                        # If the file is a .fit file, there is only one CCD
                        exptime = fits_[0].header["EXPTIME"]
                except KeyError:
                    exptime = 30
                table.add_column(exptime, name="EXPTIME")
                try:
                    table["ISO0"] = table["ISO0"].astype(np.float32)
                    table["ISO0"] = log_scale_data(table["ISO0"], 0.00001, 10000)
                    table["ELLIPTICITY"] = remove_outliers(table["ELLIPTICITY"], 0.00001, 1)
                    table["EXPTIME"] = log_scale_data(exptime * np.abs(table["BACKGROUND"] / np.mean(table["BACKGROUND"])), 0.00001, 30)
                    table["BACKGROUND"] = remove_outliers((table["BACKGROUND"] - np.mean(table["BACKGROUND"])) / np.std(table["BACKGROUND"]), -2., 2.)
                    catalog = vstack([catalog, table])
                except Exception as e:
                    print(f"Error processing table: {e}")
                    continue
        print(catalog.info())
        fits_.close()
        ldac.close()
    return catalog

def save_catalog(catalog, filename):
    """
    Save the catalog to a Parquet file.
    """
    catalog.to_pandas().to_parquet(filename, compression="gzip", engine="auto")

def main():
    # argv[1]: root_path = "/home/mike/git/computational_astro/astro_iqa"
    # argv[2]: raw_data_path = "data/raw/ngc7000"
    # argv[3]: catalog_filename = "objects_catalog_ngc7000.parquet.gz"
    # argv[4]: file_type = ".fit"
    parser = argparse.ArgumentParser(description="List all .fit files in the specified directory, generate LDAC files, process LDAC files, and save the catalog to a Parquet file.")
    parser.add_argument("root_path", type=str, help="Path to the root directory")
    parser.add_argument("raw_data_path", type=str, help="Path to the raw data directory, relative to the root directory")
    parser.add_argument("catalog_filename", type=str, help="Name of the catalog file")
    parser.add_argument("file_type", type=str, help="File type to process")

    args = parser.parse_args()
    raw_data_path = os.path.join(args.root_path, args.raw_data_path)
    
    # List all fits files in the data directory
    fits_files = list_fits_files(raw_data_path, args.file_type)
    
    # Generate LDAC files
    generate_ldac_files(raw_data_path, fits_files)
    
    # Initialize the global catalog
    catalog = Table(names=("FITS_ID", "CCD_ID", "OBJECT_ID", "ISO0", "BACKGROUND", "ELLIPTICITY", "ELONGATION", "CLASS_STAR", "FLAGS", "EXPTIME"), 
                    dtype=("S12", np.uint8, "S32", np.float32, np.float32, np.float32, np.float32, np.float32, np.int16, np.float32))
    
    # Process LDAC files and update the catalog
    catalog = process_ldac_files(raw_data_path, catalog, args.file_type)
    
    # Save the catalog
    catalog_path = os.path.join(args.root_path, "data/for_modeling")
    filename = os.path.join(catalog_path, args.catalog_filename)
    save_catalog(catalog, filename)


if __name__ == "__main__":
    main()