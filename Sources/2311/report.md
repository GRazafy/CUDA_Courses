
Open the report in Nsight Systems, leaving the previous report open for comparison.

- How does the execution time compare to that of the `addVectorsInto` kernel prior to adding asynchronous prefetching?
- Locate `cudaMemPrefetchAsync` in the *CUDA API* section of the timeline.
- How have the memory transfers changed?


#### How does the execution time compare to that of the `addVectorsInto`

After executing both programs we see that for `vector-add-no-prefetch`  the execution time is


`Time(%)      Total Time   Instances         Average         Minimum         Maximum  Name`

`100.0       104372788           1     104372788.0       104372788       104372788  addVectorsInto `


And for `vector-add-prefetch-solution` the execution time is


`Time(%)      Total Time   Instances         Average         Minimum         Maximum                                              Name`

`     100.0          506113           1        506113.0          506113             506113     addVectorsInto `

#### Locate `cudaMemPrefetchAsync` in the *CUDA API* section of the timeline.

![](../../Ressources/CUDA_API_Stat.PNG)

#### How have the memory transfers changed?
