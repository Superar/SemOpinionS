[CmdletBinding()]
param (
    [Parameter(Mandatory = $true, HelpMessage = "Path to the automatic bow file")]
    [string]
    $test,
    [Parameter(Mandatory = $true, HelpMessage = "List with all gold bow files")]
    [string[]]
    $gold
)

Write-Output "-----------ROUGE-----------"
foreach ($g in $gold) {
    $pair = "$((Get-Item $test).BaseName)-$((Get-Item $g).BaseName)"
    Write-Output $pair
    python3 -m rouge_score.rouge `
        --rouge_types=rouge1 `
        --prediction_filepattern=$test `
        --target_filepattern=$g `
        --output_filename="$((Get-Item $test).DirectoryName)\$pair.csv" `
        --noaggregate
}
