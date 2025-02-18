from ebusopt.gtfs_beb import GTFSData
from pathlib import Path

"""
Do a little pre-processing when deploying ZEBRA, so that things load
faster with the sample datasets.
"""

for agency in ['metro_may24', 'trimet_may24']:
    path_here = Path(__file__).absolute()
    gtfs_path = path_here.parent / 'ebusopt' / 'data' / 'gtfs' / agency
    gtfs = GTFSData.from_dir(dir_name=str(gtfs_path))
    all_shapes = gtfs.trips_df['shape_id'].unique().tolist()
    gtfs.summarize_shapes(all_shapes)
    gtfs.shapes_summary_df.to_csv(gtfs_path / 'shapes_summary.csv')
