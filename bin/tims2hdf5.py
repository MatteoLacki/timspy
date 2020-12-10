import argparse

from timspy.df import TimsPyDF, all_columns


P = argparse.ArgumentParser(description='Convert timsTOF Pro TDF into HDF5 with another set of parameters.',
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

AA = P.add_argument

AA('source', help='Path to the input .d folder containing TIMS data ("analysis.tdf" and "analysis.tdf_raw").')
AA('target', help='Path to the output HDF5 file.')
AA('--compression',
    help='Compression algorithm.',
    choices=('gzip','lzf','szip', 'none'),
    default='gzip')
AA('--compression_level',
    help='Compression level, valid only for gzip compression.',
    default=4,
    type=int)
AA('--chunksNo',
    help='Number of chunks. 0 for auto.',
    default=0,
    type=int)
AA('--shuffle',
    help='Perform byte shuffling to help compression.',
    action='store_true')
AA('--columns',
    help="A list of columns to be included in the output HDF5 file.",
    nargs="+",
    choices=all_columns,
    default="frame scan tof intensity")
AA('--silent', help="Do not display progress bar.", action="store_true")
args = P.parse_args()

if isinstance(args.columns, str):
    args.columns = args.columns.split()

DF = TimsPyDF(args.source)
DF.to_hdf(target_path=self.target,
          columns=args.columns,
          compression=args.compression,
          compression_level=args.compression_level,
          shuffle=args.shuffle,
          chunks=True if args.chunksNo == 0 else args.chunksNo,
          silent=args.silent)