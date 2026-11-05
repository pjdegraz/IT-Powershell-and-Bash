# D365 F&O Integration Health / Discovery Workbook

**Status: static checks complete, live Azure telemetry not validated.** This is a discovery-first Azure Monitor Workbook for the selected Application Insights resource; it does not require a separate ADX cluster.

## Import
In Application Insights → Workbooks → New → Advanced Editor → Gallery Template, paste the JSON from `D365_FO_Integration_Health.workbook.json`. Apply, select the resource, choose a known incident time and save.

## Validate the available signals first
1. Check **Data readiness** for actual records in `customEvents`, `requests`, `dependencies`, `traces`, and `exceptions`.
2. Check **All custom event names**, **Candidate integration custom events**, and **Custom event property key discovery** for the real names and field keys. The optional IntegrationTerm is a substring search, not an authoritative partner map.
3. Check **Request operation names** and **Dependency targets** for integration-related traffic. These may be empty in Microsoft-managed F&O, even when other telemetry is working.
4. Test a known incident, and compare failures with the source application and partner-side logs.

## Coverage boundaries
- **Known from Microsoft documentation:** DMF event telemetry and X++ exception telemetry when enabled.
- **Possible, not guaranteed in this F&O instance:** standard Application Insights incoming requests, outgoing dependencies, custom traces and integration-specific events.
- **Not established by this workbook:** successful partner delivery, message loss, duplicate processing, retries, Service Bus dead-letter status, or acknowledgement at the receiving system.
- **Correlation:** operation IDs only correlate if propagated across the actual emitting components.
- **Costs:** Application Insights ingestion/retention still apply. Large query time ranges may affect query costs/performance.

The committed workbook deliberately contains no live connection string, subscription ID, secrets, customer data, vendor endpoint, or production telemetry values.

## References
- https://learn.microsoft.com/en-us/dynamics365/fin-ops-core/dev-itpro/monitoring-telemetry/monitoring-available-telemetry
- https://learn.microsoft.com/en-us/dynamics365/fin-ops-core/dev-itpro/monitoring-telemetry/monitoring-developer-add-custom-signals
- https://learn.microsoft.com/en-us/azure/azure-monitor/app/data-model-request-telemetry
- https://learn.microsoft.com/en-us/azure/azure-monitor/visualize/workbooks-data-sources

**Operational check required:** Import workbook into your tenant, run the discovery panels, verify query compilation and which event types are populated before relying on health panels.