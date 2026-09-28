# excel_recalc.ps1 -Path C:\...\file.xlsx : open in (invisible) Excel, full recalculation, save, close.
param([Parameter(Mandatory=$true)][string]$Path)
$ErrorActionPreference = 'Stop'
$xl = New-Object -ComObject Excel.Application
try {
  $xl.Visible = $false; $xl.DisplayAlerts = $false
  $wb = $xl.Workbooks.Open($Path)
  $xl.CalculateFull()
  $wb.Save(); $wb.Close($true)
  Write-Output "recalculated $Path"
} finally {
  $xl.Quit(); [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($xl)
}
