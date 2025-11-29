import os
from gwdatafind import find_urls
from gwpy.timeseries import TimeSeries

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

def download_strain(event_name, detector, gps, obs_run):
    """
    Descarga strain REAL desde LIGO usando gwdatafind.
    """

    print(f"Descargando {event_name} ({detector}) — GPS {gps}")

    try:
        urls = find_urls(detector, gps, gps + 4, urltype="LOSC")
        if not urls:
            print(f"✖ No se encontraron URLs LOSC para {event_name}/{detector}")
            return None

        url = urls[0]
        print("URL encontrada:", url)

        outname = f"{event_name}_{detector}.hdf5"
        outpath = os.path.join(RAW_DIR, outname)

        print("Descargando archivo real...")
        ts = TimeSeries.read(url, format='hdf5.gwosc')
        ts.write(outpath, format='hdf5')

        print("✔ Guardado:", outpath)
        return outpath

    except Exception as e:
        print("✖ Error descargando:", e)
        return None


def load_strain(path):
    """
    Carga un archivo HDF5 real.
    """
    try:
        ts = TimeSeries.read(path, format='hdf5')
        return ts
    except Exception as e:
        print("✖ Error cargando strain:", e)
        return None
