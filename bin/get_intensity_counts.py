import argparse
import datetime
import pathlib
from pprint import pprint

from timspy.misc import folders2intensity_distribution

ap = argparse.ArgumentParser(description='Get intensity distirbutions for a given set of conditions.',
                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ARG = ap.add_argument

ARG('folders',
    nargs='+',
    help="Path(s) to a timsTOF .d folder(s) containing 'analysis.tdf' and 'analysis.tdf_raw'.",
    type=pathlib.Path)

ARG('--output_folder',
    help="Path to output folder to save output to.",
    default=".",
    type=pathlib.Path)

ARG("--verbose",
    help="Show CLI messages.",
    action='store_true')

ARG("--debug",
    help="Run in debug: 10 first frames.",
    action='store_true')


args = ap.parse_args()

analysis_time = datetime.datetime.now().strftime('%d_%m_%Y__%H_%M_%S')
output = f"timstof_intensity_counts__{analysis_time}.csv"

assert all(f.exists() for f in args.folders), "Datasets you want to analyze are missing!"
if args.verbose:
    print("Analysing folders:")
    for f in args.folders:
        print(f)
    print()

intensity_counts_df = folders2intensity_distribution(args.folders, 
                                                     frame_numbers=list(range(1, 10)) if args.debug else None, 
                                                     verbose=args.verbose)
if args.verbose:
    print("Saving")
intensity_counts_df.to_csv(output)

if args.verbose:
    print("Thanks!")
