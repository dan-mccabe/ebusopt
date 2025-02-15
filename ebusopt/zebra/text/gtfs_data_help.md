### Overview
On this page, you can provide the data files describing the transit system you\'re looking at serving with BEBs. You can use the default files we provide in order to check how the app functions and explore results for King County Metro and TriMet, the transit agencies in the Seattle and Portland areas, respectively. For other locations, please upload GTFS files manually.

Even if you are not familiar with GTFS, locating GTFS files for your agency and uploading them here should be easy, as they are published by most American transit agencies. For example, here are links to the GTFS files for [CTtransit](https://www.cttransit.com/about/developers) in Connecticut and [CapMetro](https://data.austintexas.gov/widgets/r4v4-vz24) in Austin, Texas. After downloading these files yourself, you can upload them to ZEBRA in the **:violet[📄 Select Data]** tab. Advanced users can also use custom GTFS files as desired, as long as they meet the criteria listed under **Data Requirements** below.

### Instructions
Use the first dropdown menu to specify whether you'd like to use the default GTFS files or upload your own. Once the files are uploaded, use the date input to select a single date as the basis of your analysis. We need to tie our analysis to a particular date because bus assignments (a.k.a. *blocks*) change from day to day, and our analysis centers on the blocks defined in GTFS that let us calculate how far every bus needs to drive on a given day. By default, we pick the day with the greatest number of trips to serve -- the "hardest" day on the calendar.

### Data Requirements
ZEBRA relies on data in the [General Transit Feed Specification (GTFS)](https://developers.google.com/transit/gtfs/reference) format. Below, we list the necessary tables that our analysis relies on as well as the specific columns that must be present. Note that **GTFS defines some optional columns that are required for this tool**, so not every agency's GTFS feed is complete enough to be used.

- At least one of `calendar.txt` and `calendar_dates.txt` must be supplied with all fields. This is the same as the requirement for standard GTFS.
- `trips.txt` is required. It must contain the fields `trip_id`, `route_id`, `service_id`, `block_id`, and `shape_id`.
- `routes.txt` is required. It must contain the fields `route_id`, `route_short_name`, and `route_type`.
- `shapes.txt` is required and must contain the fields `shape_id`, `shape_pt_lat`, `shape_pt_lon`, and `shape_pt_sequence`. Previous versions of ZEBRA also required `shape_dist_traveled`, but the app has been updated to calculate this distance manually and this field is no longer used.
- `stop_times.txt` is required with fields `trip_id` and `arrival_time`.