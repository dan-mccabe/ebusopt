### Overview
Here, you can optionally specify which bus routes to include in the analysis. By default, all routes are included. You may want to exclude some routes if they will not be served by BEBs (e.g., trolleybus), or include only a subset of routes planned for more immediate BEB deployment.

### Instructions
Use the selection box to specify which routes are included. Click the small **x** next to a route name to remove it, or click the gray **⊗** on the right of the selection box to clear all routes and start over. Place your cursor inside the selection box to see all routes that can be chosen, and type to filter the list down.

### How It Works
The list of route choices is based on the `route_short_name` GTFS field in `routes.txt`. All blocks that include *any* trips on at least one of the selected routes will be included in the analysis on the following pages. That is, if you select route A but not B, a block that completes trips on both route A and route B *will* be included.