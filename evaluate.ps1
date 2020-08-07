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
    Write-Output "$(Split-Path -Path $test -Leaf) -- $(Split-Path -Path $g -Leaf)"
    python3 .\venv\Scripts\smatch.py -f $test $g
}

Write-Output "------------SEMA------------"
foreach ($g in $gold) {
    Write-Output "$test--$g"
    python3 .\venv\sema\sema.py -t $test -g $g
}
Write-Output "----------------------------"
