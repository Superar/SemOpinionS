[CmdletBinding()]
param (
    [Parameter(Mandatory = $true, HelpMessage = "Path to the automatic summary graph file")]
    [string]
    $test,
    [Parameter(Mandatory = $true, HelpMessage = "List with all gold graph files")]
    [string[]]
    $gold
)

Write-Output "-----------SMATCH-----------"
foreach ($g in $gold) {
    Write-Output "$((Get-Item $test).BaseName)-$((Get-Item $g).BaseName)"
    python3 .\venv\Scripts\smatch.py -f $test $g
}

Write-Output "------------SEMA------------"
foreach ($g in $gold) {
    Write-Output "$((Get-Item $test).BaseName)-$((Get-Item $g).BaseName)"
    python3 .\venv\sema\sema.py -t $test -g $g
}
Write-Output "----------------------------"
