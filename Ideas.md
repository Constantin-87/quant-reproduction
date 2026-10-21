# Ideas for experimentation

## Are all levels needed?
- I question whether all quantiles for depths 0 - d are needed
- Are early depths (0, 1) needed? Since these are almost equivelant to subsampled and sorted version of the original datasets, basically providing only context on the distribution, spread etc. and not of the shape itself
- Are later depths (approaching d) needed? These will effectivley become the original dataset itself as the number of intervals approaches the number of values
- Do intervals for all depth values (particularly for large d values) need to be taken? Or can every second/third depth be taken?
- Do both normal and shift versions need to be calculated for each level. Particularly in later levels, might these not be similar? (they may provide context to eachother though)
- Would need to evaluate accuracy vs complexity. Since removing features will most likey reduce accuracy, the question is whether the reduced complexity is worth it, and by how much accuracy drops

## Are features other than interval quantiles relevant?
- Specifically, can quantile information be replaced with mean, varience, range and other statisical information
- Particularly for lower depths (0 - 5) and larger datasets, this could massively reduce the number of features
- This would also provide more straight forward representations of the same information
- Might be less relevant for first and second differences
- Again, if this results in fewer features, accuracy will likely drop, so watch for accuracy - complexity trade off

## Are Inter-intnerval features useful?
- For instances, the varience, differences, or distances between intervals
- Would provide context between intervals (granted, taking measures of different depths and shift quantiles already do this)
- Information could be calculated for neighboring or parent-child intervals
- This can be tested skipped depth counts, since this would hopefully reflect the information level by level depths would provide

## Are different representaitons useful?
- Like different convolutions or lag difference done my Constantin
- Smoothing, spike detection etc.
- Need to be carful with this. Since this technique largely focuses on minimal complexity adding such represenations could increase complexity to an undesireable degree. More minimal convoluitions would be better (length 3 - 5).
- Could replace existing features (different methods of spike detection can replace second differencing)

## Can dynamic intervals be used?
- This is mentioned by the paper
- My specific though on this is that in the original version of quant, information about changes in direction are lost, particularly at lower depth intervals
- Intervals could be made based on changes in direction (although data would likely need to be smoothened)
- Intervals in this setting could have different sizes. As long as the number of quantile values per depth is the same, the algorithm will continue to work
- Alternatively, this could be combined with summary features (mean, varience etc to replace quantiles), in which case interval length doesnt matter, so long as there are the same number for the corrosponding depth
- This however causes issues with data representation. Normal quant works in absolute time distances. The time between measures is constant. In this varient, the time between measures is highly variable. Measures to deal with this would be needed (in most cases)