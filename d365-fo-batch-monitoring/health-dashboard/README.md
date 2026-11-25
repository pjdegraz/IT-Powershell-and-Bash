# D365 F&O Health Dashboard (no added cost)

**Status: structure-checked only, not validated against live telemetry.** The workbook JSON was generated and checked locally (valid JSON, every parameter it references is defined). None of the queries have been run against a real Application Insights resource yet.

A single Azure Monitor workbook for the morning check on Dynamics 365 Finance and Operations (F&O): **is anything wrong right now, which feed, and why?** It sits on top of the investigation workbooks in this folder and links out to them, rather than replacing them.

It only reads telemetry that is already in your Application Insights resource, plus optional Azure Service Bus platform metrics. It creates no alert rules, turns on no new telemetry categories and creates no billable resources.

## What's on it

| # | Section | Source | What it answers |
| --- | --- | --- | --- |
| 1 | Headline tiles | Application Insights | One tile per configured feed (hours since last finished import), plus 24-hour counts of batch failures, Data Management Framework (DMF) errors and X++ exceptions |
| 2 | Feed freshness | Application Insights | For each DMF data project: last finished import, hours since then, errors in 24 hours, OK or Check |
| 2 | DMF errors | Application Insights | Errors grouped by data project, entity and category, with a sample message |
| 3 | DMF import heartbeat | Application Insights | Imports started, finished and errored per hour over 48 hours |
| 3 | Batch failures by job | Application Insights | Failed batch tasks by job caption and class, with a sample error |
| 4 | Store sync proxy | Application Insights | Batch activity for Commerce distribution, P-job, order sync and statement jobs (inferred, see below) |
| 4 | Service Bus queue health | Service Bus platform metrics | Active, dead-lettered, incoming and outgoing messages (optional) |
| 5 | X++ exceptions | Application Insights | Top 15 exception messages in 24 hours |
| 5 | Batch capacity | Application Insights | Active vs queued batch tasks over 48 hours |
| 5 | Telemetry volume | Application Insights | Records per day by table, as a cost guard |
| 5 | Where to look next | Links | Lifecycle Services (LCS) and F&O pages that hold the authoritative record |

The idea behind the headline is that a feed which quietly stops is worse than one that fails loudly. A failure raises a DMF error; a stopped hourly import raises nothing. "Last finished 5 hours ago" on an hourly feed is the most useful thing this workbook can tell you.

## Import

1. Open the Application Insights resource that receives F&O telemetry in the Azure portal.
2. Go to **Workbooks > New > Edit > Advanced Editor**.
3. In **Gallery Template** mode, paste the content of `D365_FO_Health_Dashboard.workbook.json`, apply, and save it as **D365 F&O Health Dashboard**.
4. Fill in the setup bar (next section).
5. Optional: pin the headline tiles and the Service Bus chart to an Azure portal dashboard (each workbook item has a pin option in edit mode).

Saving a shared workbook or dashboard stores it as an Azure resource in a resource group, so you need a role that can write there (typically Workbook Contributor or Contributor on that resource group). Neither one bills.

## Setup bar

| Parameter | What to enter |
| --- | --- |
| Application Insights resource | The resource receiving F&O telemetry |
| Lookback window | Defaults to 7 days. Feed freshness looks across the whole window; the other panels use fixed 24 or 48 hour windows inside it |
| Feed schedule | `ProjectName=MaxHours;OtherProject=MaxHours`. Use each DMF data project's name **after** the trailing GUID is stripped (the same rule the DMF FastTrack workbook uses), and the most hours it may go between finished imports. An hourly import is usually `2` (one hour plus slack). A configured project that sent nothing in the window still shows up as a red row instead of disappearing |
| Store sync batch job caption terms | Words found in the captions of your Commerce distribution schedule, P-job, order synchronization and statement posting batch jobs, separated by `;` |
| DMF event names | Start, end and error event names. Defaults follow the FastTrack samples (`DMFImportJobStart`, `DMFImportJobEnd`, `DMFError`). Change them here if your environment emits different names |
| Service Bus namespace | Optional. Pick the namespace that carries an integration's messages to show its queue chart |

The committed workbook contains placeholder project names only: no connection strings, subscription IDs, resource IDs, partner names or production values.

## Validate before trusting it

- [ ] **Event names.** Run *All custom event names* in the `D365_FO_Integration_Health` workbook. Microsoft's telemetry documentation lists DMF signals by display name ("DMFJob Start", "DMFJob End", "DMFJob Status"), which may not match the FastTrack names. Put the real names in the setup bar.
- [ ] **Build version.** Microsoft lists minimum builds for these signals: batch telemetry from 10.0.44/10.0.45, DMF telemetry from 10.0.45 to 10.0.47 depending on the update, batch job occurrence from 10.0.48. Check the environment version in Lifecycle Services (LCS).
- [ ] **Categories switched on.** Check **System administration > Setup > Monitoring and telemetry parameters** in F&O for batch and DMF.
- [ ] **A finish means success.** Confirm whether the DMF end event also fires for failed runs. If it does, the freshness check needs a status field from the event to tell a good finish from a bad one.
- [ ] **Field names.** Confirm `DataProjectName`, `EntityName`, `ErrorMessage`, `ErrorCategory`, `BatchJobCaption`, `BatchJobId`, `ClassName`, `ExceptionMessage` and `BatchThreadInfo`'s `InfoMessage` are populated (use the property key discovery panel in the Integration Health workbook).
- [ ] **Store sync captions.** Find the real captions of the Commerce jobs in **System administration > Inquiries > Batch jobs** and check they appear in `BatchJobCaption`. If they don't, ignore that panel and use the F&O pages in the links section.
- [ ] **Sampling and retention.** Sampling can make a feed look quieter than it is. Check the sampling rate and retention on the Application Insights resource.
- [ ] **One known incident.** Pick a day a feed failed and confirm the workbook shows it the way F&O's own job history does.
- [ ] **Service Bus metric names.** The metrics item uses `ActiveMessages`, `DeadletteredMessages`, `IncomingMessages` and `OutgoingMessages`. If the chart is empty, re-add them through the metric picker in edit mode.

## Limitations

- **Store sync is an inferred proxy.** F&O telemetry has no Commerce-specific signal. The panel assumes the jobs that move data between headquarters and stores run as F&O batch jobs with recognizable captions. Whether data actually reached the stores lives in F&O at **Retail and Commerce > Inquiries and reports > Commerce Data Exchange > Download sessions** and **Upload sessions**.
- **Green is not proof of delivery.** Telemetry shows what F&O did, not what a partner system received.
- **The volume chart counts records, not billed gigabytes.** For gigabytes, use the resource's **Usage and estimated costs** page or run this against the Log Analytics workspace behind Application Insights:

```kusto
AppEvents
| where TimeGenerated > ago(30d) and _IsBillable =~ "true"
| summarize BilledGB = round(sum(_BilledSize) / 1073741824.0, 3) by bin(TimeGenerated, 1d), Name
| render columnchart
```

- **Times** are stored in UTC; grid columns ending in `ET` are converted to `America/New_York`.

## What could add cost (and is deliberately left out)

- Alert rules (log search alerts are billed per rule per month).
- Turning on more F&O telemetry categories, or lowering sampling (more ingestion).
- Sending Service Bus diagnostic logs to Log Analytics (the free platform metrics cover the chart).

## References

- F&O available telemetry and version prerequisites: https://learn.microsoft.com/en-us/dynamics365/fin-ops-core/dev-itpro/monitoring-telemetry/monitoring-available-telemetry
- Azure Workbooks data sources: https://learn.microsoft.com/en-us/azure/azure-monitor/visualize/workbooks-data-sources
