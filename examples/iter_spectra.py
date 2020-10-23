import numpy as np
import pandas as pd
from pathlib import Path

from timspy.timspy import TimsDIA

D = TimsDIA('/mnt/samsung/bruker/BrukerMIDIA/MIDIA_CE10_precursor/20190912_HeLa_Bruker_TEN_MIDIA_200ng_CE10_100ms_Slot1-9_1_488.d')

def iter_spectra(D, frames, max_scan):
    for f in frames:
        for scanNo, tof, intensities in D.iterScans(f, 0, max_scan):
            yield f, scanNo, D.tof2mz(tof, f), intensities

D.min_frame
D.max_frame
list(iter_spectra(D, [100], 918))
D.frames