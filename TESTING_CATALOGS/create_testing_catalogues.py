import os
from astropy.table import Table
from astropy.io import fits
import numpy as np


def create_example_fits_catalogue(filename="example.fits", num_rows=100):
    """
    Creates an example FITS catalogue with astronomical data.

    Args:
        filename (str): The name of the FITS file to create.
        num_rows (int): The number of rows (sources) in the catalogue.
    """
    ra = np.random.uniform(0, 360, num_rows)
    dec = np.random.uniform(-90, 90, num_rows)
    flux_g = np.random.lognormal(mean=2, sigma=0.5, size=num_rows) * 100
    flux_r = np.random.lognormal(mean=2.1, sigma=0.6, size=num_rows) * 100
    flux_i = np.random.lognormal(mean=2.2, sigma=0.7, size=num_rows) * 100
    size_maj = np.random.uniform(0.5, 5.0, num_rows)
    size_min = np.random.uniform(0.3, 3.0, num_rows)
    pa = np.random.uniform(0, 180, num_rows)

    # Create an Astropy Table
    data = Table()
    data["RA"] = ra
    data["DEC"] = dec
    data["FLUX_G"] = flux_g
    data["FLUX_R"] = flux_r
    data["FLUX_I"] = flux_i
    data["SIZE_MAJ"] = size_maj
    data["SIZE_MIN"] = size_min
    data["POS_ANGLE"] = pa

    # Write the table to a FITS file
    data.write(filename, format="fits", overwrite=True)
    print(f"Created example FITS catalogue: {filename} with {num_rows} rows.")


if __name__ == "__main__":
    # Ensure the output directory exists
    output_dir = "TESTING_CATALOGS"
    os.makedirs(output_dir, exist_ok=True)

    # Create example FITS catalogue
    create_example_fits_catalogue(os.path.join(output_dir, "input.fits"), num_rows=100)

    print("Example FITS catalogue created successfully.")
    # You can add more functions to create other catalogues as needed.
