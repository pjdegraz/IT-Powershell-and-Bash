# D365 F&O – Batch Incident Explorer (Hierarchy Edition)

## What changed
The main investigation view is now an **expandable Azure Monitor Workbook parent/child tree**:

    JOB 12345 (logical job ID, summary of observed task executions)
      TASK 9901 · MyFirmingClass
        RUN 2026-11-04 13:02 ET · abcdef123456 — Finished
        RUN 2026-11-04 13:17 ET · abcdef987654 — No finish observed
      TASK 9902 · AnotherClass
        RUN 2026-11-04 13:03 ET · 555555123456 — Finished

This is an **illustrative layout**, not real production data. The three levels are logical batch **job** → batch **task** → correlated task **execution**. `activityId` associates each start/finish/failure event with a task execution, following Microsoft's FastTrack KQL example. The tree is configured through Azure Workbooks' native `gridSettings.hierarchySettings` (`treeType: 0`, `idColumn: Id`, `parentColumn: ParentId`, `expanderColumn: Name`).

## Import
1. Open the Application Insights resource receiving D365 telemetry in the Azure portal.
2. Go to **Workbooks → New → Edit → Advanced Editor**.
3. In **Gallery Template** mode paste the content of `D365_FO_Batch_Incident_Explorer_Hierarchy.workbook.json`. Apply and save as **D365 F&O Batch Incident Explorer — Hierarchy**.
4. Choose your **Application Insights resource** in the workbook parameter selector.
5. Choose Custom date/time such as **1:00–2:00 PM Eastern**, leave Batch Job ID blank, expand a JOB and then a TASK. The time picker uses your browser/portal local timezone; the query display converts to `America/New_York`.
6. Use **Telemetry event schema / sample properties** near the bottom to check that your environment emits the required fields before trusting the tree. If the tree is empty, check those samples and the no-activityId diagnostics.

## Configuration and field requirements
- Turn on **Batch start time**, **Batch stop time**, **Batch failure** in F&O Monitoring and Telemetry; enable batch job occurrence events if supported by your F&O build. The tree uses task start/finish/failure events (`BatchTaskStart`, `BatchTaskFinished`, `BatchTaskFailure`) in Application Insights `customEvents`, not direct SQL access or a custom F&O agent.
- `customDimensions.activityId` is required to correlate each task execution. `customDimensions.BatchJobId`, `BatchJobTaskId` or `BatchTaskId` are used to build the JOB and TASK parent keys. When a task ID is missing, the hierarchy uses the activity ID as a **nonmerged fallback task**, which is more accurate than merging unrelated tasks.
- This update fixes the task ID fallback in the original template: using `iif(isnotempty(...))` instead of `coalesce(...)` for empty strings.
- The record correlation and column settings are based on published examples, and the workbook JSON was structure-checked locally, **not tested against your live telemetry**.

## Important limitations
- **Parent JOB is not one scheduled job occurrence.** Repeated scheduled executions with the same BatchJobId appear under the same job, with separate child RUN rows under their task IDs. F&O occurrence events are a different signal; mapping a task to a precise parent occurrence must be validated against *actual occurrence correlation IDs* in your environment. The raw occurrence telemetry panel remains in the template for this purpose.
- Job/task node start/end columns show the first observed task start and last observed finish **across matching executions**, not the full parent job's guaranteed state or duration.
- **No finish observed is not the same as confirmed running or stuck.** Sampling, dropped events, or delayed delivery can cause that condition. Check the F&O batch history for authoritative status.
- To inspect all competing jobs between 1 PM and 2 PM, retain task telemetry from those jobs. Filtering out noncritical jobs before Application Insights ingestion creates blind spots. The Job ID filter restricts the query view only, not collection cost.
- Lookback into events earlier than the incident window defaults to 2 days (adjustable 1–7 days), so unusually long processes may require more history. Ingestion limits/retention can prevent complete reconstruction. The table row limit is 10,000. Very high volume may require narrowing the window.
- No alerts or F&O job schedules are created or changed by the workbook.

## References
- F&O batch telemetry signals/version prerequisites: https://learn.microsoft.com/en-us/dynamics365/fin-ops-core/dev-itpro/monitoring-telemetry/monitoring-available-telemetry
- Microsoft FastTrack example: https://community.dynamics.com/blogs/post/?postid=99ccc3c5-fb78-f011-b4cb-7c1e52649b7e
- Azure Monitor native parent/child tree: https://learn.microsoft.com/en-us/azure/azure-monitor/visualize/workbooks-tree-visualizations
- Azure Workbooks tree JSON examples: https://github.com/microsoft/Application-Insights-Workbooks/blob/master/Workbooks/Azure%20Monitor%20-%20Applications/Performance%20Analysis/Performance%20Analysis.workbook