### Overview
The **:violet[📊 Block Distances]** tab shows two different visualizations of the total distance every bus in your system must drive each day. The histogram on the left plots the number of buses whose daily range lies within a certain range, while the empirical CDF on the right plots the total block distance on the horizontal axis and the percentage of blocks with total distance less than or equal to that value on the vertical axis. The **:violet[🔋 Range Requirements]** tab shows how much more range each bus in your system would need to make all blocks completable with depot charging only.

### Instructions
Use the numeric input at the top of the page to set the estimated range of your buses in miles. On the **:violet[📊 Block Distances]** tab, you can move your cursor over the charts to see more details. The same applies to the combined CDF/histogram chart on the **:violet[🔋 Range Requirements]** tab. Here, you can also see how the results change as you alter the bus range. As soon as you enter a new range value, the text information and interactive chart will update with the new results. 

### Interpreting Results
The results shown on this page help illustrate what ranges (and correspondingly, battery sizes) will be appropriate for the buses in your analysis. The histogram under **:violet[📊 Block Distances]** is straightforward to interpret, with the height of each bar corresponding to the number of blocks whose total distance lies within the distance bin on the x-axis. The farther left on the chart a bar is, the less range is required to complete the corresponding blocks.

The line charts shown on both tabs might be less familiar, but are quite useful for assessing range needs. These plots are [Empirical CDFs](https://en.wikipedia.org/wiki/Empirical_distribution_function) of the same block distance data. For a given distance on the x-axis, the value of the line on the y-axis gives the **percentage of blocks whose total distance is less than or equal to the x-value**. Put another way, the y-value tells us how many blocks could be completed without opportunity charging with buses whose range is equal to the x-value. If we hover over the chart and see **Distance (miles)=150, percent=80**, this means that 80% of blocks have a total distance of 150 miles or less.

### How It Works
We process the GTFS data to calculate the total in-service driving distance for every `block_id` that is active on the given date of service. This process consists of a few separate steps:

1) For every `shape_id` that applies to the selected date and routes, we calculate its distance based on the shape's coordinates as defined in `shapes.txt`. We determine the distance between each pair of consecutive points using the Haversine formula and add these up to get the total distance. We also identify the coordinates of the first and last point in each shape.
2) We match each active trip to its corresponding shape. This gives us the trip's total distance and start/end points as determined in Step 1.
3) For each block, we sort the trips in order of start time. We then add the estimated deadhead distance between all pairs of trips in this block based on the Manhattan distance. We use Manhattan distance rather than exact driving distance in order to get a decent estimate quickly.
4) We add up all the service and deadhead miles for each block to determine its total distance.

One important limitation to note is that **ZEBRA does not incorporate the distances of pull-out/pull-in trips from/to bus depots**. We neglect this distance because GTFS does not provide information about depot locations, or about which depot a block originates from. Users should be aware that these additional deadhead trips may significantly increase daily range needs.

#