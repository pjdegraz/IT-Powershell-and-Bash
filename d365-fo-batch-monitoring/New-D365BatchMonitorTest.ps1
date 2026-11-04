#Requires -Modules Az.Accounts, Az.Resources, Az.OperationalInsights, Az.ApplicationInsights
<#
.SYNOPSIS
  Provision a D365 F&O UAT Application Insights resource in an EXISTING Log Analytics
  workspace by default; optionally list workspaces or create a new workspace.

.DESCRIPTION
  REUSE mode never changes an existing Log Analytics workspace: no retention,
  ingestion cap, SKU, or table settings are modified. Only a dedicated
  Application Insights resource is created (or verified).

  BROWSE mode lists workspaces without creating or changing anything.

  CREATE mode is explicit (-CreateWorkspace). Only a newly created workspace
  receives 30-day retention. Existing workspaces are never reconfigured.

  For a newly created Application Insights resource, a 0.10 GB/day APPLICATION
  INSIGHTS cap is applied. An existing Application Insights resource is not
  modified. A daily cap is a safeguard, NOT a monthly 5-GB billing hard limit.
  When a cap is reached, collection can stop and events may be lost; actual
  ingested volume can exceed the cap. A workspace cap can also affect other
  applications, so this script does not set one on shared workspaces.

  This script does not modify D365 F&O or Microsoft-managed resources.

.EXAMPLE
  # 1. Browse workspaces in the target subscription.
  .\New-D365BatchMonitorTest.ps1 -SubscriptionId '<subscription-guid>' -ListWorkspaces

.EXAMPLE
  # 2. Reuse an existing workspace (recommended for UAT).
  .\New-D365BatchMonitorTest.ps1 -SubscriptionId '<subscription-guid>' -WorkspaceResourceGroupName 'rg-monitoring' -WorkspaceName 'law-shared-uat' -AppInsightsResourceGroupName 'rg-uat' -ApplicationInsightsName 'appi-d365fo-uat-batch'

.EXAMPLE
  # 3. Create a workspace, ONLY when explicitly requested.
  .\New-D365BatchMonitorTest.ps1 -SubscriptionId '<subscription-guid>' -CreateWorkspace -WorkspaceResourceGroupName 'rg-uat' -WorkspaceName 'law-d365fo-uat-test' -AppInsightsResourceGroupName 'rg-uat' -ApplicationInsightsName 'appi-d365fo-uat-batch' -Location 'eastus'

.NOTES
  Install required Azure PowerShell modules first:
  Install-Module Az.Accounts,Az.Resources,Az.OperationalInsights,Az.ApplicationInsights -Scope CurrentUser

  Requires resource-reader access to browse and permission to create Application
  Insights in the target resource group. Never commit connection strings or
  live subscription IDs to a public repository.
#>
[CmdletBinding(DefaultParameterSetName='Reuse')]
param(
    [Parameter(Mandatory=$true)]
    [string]$SubscriptionId,

    [Parameter(Mandatory=$true, ParameterSetName='Browse')]
    [switch]$ListWorkspaces,

    [Parameter(Mandatory=$true, ParameterSetName='Reuse')]
    [Parameter(Mandatory=$true, ParameterSetName='Create')]
    [string]$WorkspaceResourceGroupName,

    [Parameter(Mandatory=$true, ParameterSetName='Reuse')]
    [Parameter(Mandatory=$true, ParameterSetName='Create')]
    [string]$WorkspaceName,

    [Parameter(Mandatory=$true, ParameterSetName='Create')]
    [switch]$CreateWorkspace,

    [Parameter(Mandatory=$true, ParameterSetName='Reuse')]
    [Parameter(Mandatory=$true, ParameterSetName='Create')]
    [string]$AppInsightsResourceGroupName,

    [Parameter(ParameterSetName='Reuse')]
    [Parameter(ParameterSetName='Create')]
    [string]$ApplicationInsightsName = 'appi-d365fo-uat-batch',

    [Parameter(ParameterSetName='Create')]
    [string]$Location = 'eastus',

    [Parameter(ParameterSetName='Create')]
    [ValidateRange(30,730)]
    [int]$WorkspaceRetentionDays = 30,

    [Parameter(ParameterSetName='Reuse')]
    [Parameter(ParameterSetName='Create')]
    [ValidateRange(0.023,1000)]
    [double]$NewAppInsightsDailyCapGb = 0.10
)

$ErrorActionPreference = 'Stop'

Import-Module Az.Accounts
Import-Module Az.Resources
Import-Module Az.OperationalInsights
Import-Module Az.ApplicationInsights

if (-not (Get-AzContext)) {
    Connect-AzAccount | Out-Null
}
Set-AzContext -SubscriptionId $SubscriptionId | Out-Null

if ($PSCmdlet.ParameterSetName -eq 'Browse') {
    Get-AzOperationalInsightsWorkspace |
        Select-Object ResourceGroupName, Name, Location, Sku, RetentionInDays,
            @{Name='ResourceId';Expression={$_.ResourceId}} |
        Sort-Object ResourceGroupName, Name
    return
}

# Do not implicitly create resource groups in a managed Azure tenant.
$workspaceRg = Get-AzResourceGroup -Name $WorkspaceResourceGroupName -ErrorAction SilentlyContinue
if (-not $workspaceRg) {
    throw "Workspace resource group '$WorkspaceResourceGroupName' not found. No changes were made."
}
$appRg = Get-AzResourceGroup -Name $AppInsightsResourceGroupName -ErrorAction SilentlyContinue
if (-not $appRg) {
    throw "Application Insights resource group '$AppInsightsResourceGroupName' not found. No changes were made."
}

$workspace = Get-AzOperationalInsightsWorkspace -ResourceGroupName $WorkspaceResourceGroupName -Name $WorkspaceName -ErrorAction SilentlyContinue

if ($PSCmdlet.ParameterSetName -eq 'Create') {
    if ($workspace) {
        throw "Workspace '$WorkspaceName' already exists. Use reuse mode (omit -CreateWorkspace) to protect existing settings."
    }
    $workspace = New-AzOperationalInsightsWorkspace -ResourceGroupName $WorkspaceResourceGroupName -Name $WorkspaceName -Location $Location -Sku PerGB2018 -RetentionInDays $WorkspaceRetentionDays
    Write-Host ("Created NEW Log Analytics workspace: {0}" -f $WorkspaceName)
}
elseif (-not $workspace) {
    throw "Workspace '$WorkspaceName' not found in '$WorkspaceResourceGroupName'. No changes were made. Use -ListWorkspaces to browse."
}
else {
    Write-Host ("Reusing workspace '{0}' (retention, limits and settings are untouched)." -f $WorkspaceName)
}

$workspaceResourceId = '/subscriptions/{0}/resourceGroups/{1}/providers/Microsoft.OperationalInsights/workspaces/{2}' -f $SubscriptionId, $WorkspaceResourceGroupName, $WorkspaceName

$appInsights = Get-AzApplicationInsights -ResourceGroupName $AppInsightsResourceGroupName -Name $ApplicationInsightsName -ErrorAction SilentlyContinue
$createdNewAppInsights = $false

if ($appInsights) {
    $linkedId = [string]$appInsights.WorkspaceResourceId
    if ([string]::IsNullOrWhiteSpace($linkedId) -or
        ($linkedId.TrimEnd('/') -ine $workspaceResourceId.TrimEnd('/'))) {
        throw "Application Insights '$ApplicationInsightsName' already exists but is not linked to this workspace. It was NOT changed. Choose another name."
    }
    Write-Host ("Reusing existing Application Insights '{0}' (existing cap untouched)." -f $ApplicationInsightsName)
}
else {
    # For a workspace-based component, use the location of its Log Analytics workspace.
    $appInsights = New-AzApplicationInsights -ResourceGroupName $AppInsightsResourceGroupName -Name $ApplicationInsightsName -Location $workspace.Location -ApplicationType other -WorkspaceResourceId $workspaceResourceId
    $createdNewAppInsights = $true
    try {
        Set-AzApplicationInsightsDailyCap -ResourceGroupName $AppInsightsResourceGroupName -Name $ApplicationInsightsName -DailyCapGB $NewAppInsightsDailyCapGb | Out-Null
    }
    catch {
        throw ("Application Insights '{0}' was created, but its daily cap could NOT be configured. Do not connect F&O telemetry until the cap is reviewed in the Azure portal. Error: {1}" -f $ApplicationInsightsName, $_.Exception.Message)
    }
    Write-Host ("Created Application Insights '{0}' with intended daily cap {1} GB/day." -f $ApplicationInsightsName, $NewAppInsightsDailyCapGb)
}

[pscustomobject]@{
    SubscriptionId               = $SubscriptionId
    WorkspaceResourceGroup       = $WorkspaceResourceGroupName
    WorkspaceName                = $WorkspaceName
    WorkspaceResourceId          = $workspaceResourceId
    WorkspaceSettingsModified    = ($PSCmdlet.ParameterSetName -eq 'Create')
    ApplicationInsightsName      = $ApplicationInsightsName
    AppInsightsResourceGroup     = $AppInsightsResourceGroupName
    CreatedNewApplicationInsights= $createdNewAppInsights
    NewAppInsightsDailyCapGB     = $(if ($createdNewAppInsights) { $NewAppInsightsDailyCapGb } else { 'Not changed; inspect in portal' })
    Nominal31DayIngestionGB      = $(if ($createdNewAppInsights) { [math]::Round($NewAppInsightsDailyCapGb * 31,2) } else { 'Not calculated' })
}

Write-Warning 'An Application Insights daily cap does NOT guarantee a fixed monthly charge or protect other data in a shared workspace. Ingestion can exceed the cap. The cap can also cause missing telemetry.'
Write-Host 'Next: Application Insights > Overview > Connection string. Paste the string into F&O UAT Monitoring and Telemetry parameters. Do NOT commit the string to GitHub.'
Write-Host 'To monitor total workspace billable ingestion, query the workspace Logs:'
Write-Host 'Usage | where TimeGenerated > ago(31d) and IsBillable == true | summarize GB = sum(Quantity)/1000.0'
