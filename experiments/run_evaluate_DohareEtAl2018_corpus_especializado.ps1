Write-Host "...........Galaxy-SIII..........." -ForegroundColor red
.\evaluation\evaluate_amr.ps1 `
    -test .\out\Galaxy-SIII\DohareEtAl2018.amr `
    -gold .\out\Galaxy-SIII\SumarioExtrativo_1.amr, .\out\Galaxy-SIII\SumarioExtrativo_2.amr, .\out\Galaxy-SIII\SumarioExtrativo_3.amr, .\out\Galaxy-SIII\SumarioExtrativo_4.amr, .\out\Galaxy-SIII\SumarioExtrativo_5.amr

.\evaluation\evaluate_bow.ps1 `
    -test .\out\Galaxy-SIII\DohareEtAl2018.bow `
    -gold .\out\Galaxy-SIII\SumarioExtrativo_1.bow, .\out\Galaxy-SIII\SumarioExtrativo_2.bow, .\out\Galaxy-SIII\SumarioExtrativo_3.bow, .\out\Galaxy-SIII\SumarioExtrativo_4.bow, .\out\Galaxy-SIII\SumarioExtrativo_5.bow

Write-Host "...........Iphone-5..........." -ForegroundColor red
.\evaluation\evaluate_amr.ps1 `
    -test .\out\Iphone-5\DohareEtAl2018.amr `
    -gold .\out\Iphone-5\SumarioExtrativo_1.amr, .\out\Iphone-5\SumarioExtrativo_2.amr, .\out\Iphone-5\SumarioExtrativo_3.amr, .\out\Iphone-5\SumarioExtrativo_4.amr, .\out\Iphone-5\SumarioExtrativo_5.amr


.\evaluation\evaluate_bow.ps1 `
    -test .\out\Iphone-5\DohareEtAl2018.bow `
    -gold .\out\Iphone-5\SumarioExtrativo_1.bow, .\out\Iphone-5\SumarioExtrativo_2.bow, .\out\Iphone-5\SumarioExtrativo_3.bow, .\out\Iphone-5\SumarioExtrativo_4.bow, .\out\Iphone-5\SumarioExtrativo_5.bow

Write-Host "...........O-Apanhador-no-Campo-de-Centeio..........." -ForegroundColor red
.\evaluation\evaluate_amr.ps1 `
    -test .\out\O-Apanhador-no-Campo-de-Centeio\DohareEtAl2018.amr `
    -gold .\out\O-Apanhador-no-Campo-de-Centeio\SumarioExtrativo_1.amr, .\out\O-Apanhador-no-Campo-de-Centeio\SumarioExtrativo_2.amr, .\out\O-Apanhador-no-Campo-de-Centeio\SumarioExtrativo_3.amr, .\out\O-Apanhador-no-Campo-de-Centeio\SumarioExtrativo_4.amr, .\out\O-Apanhador-no-Campo-de-Centeio\SumarioExtrativo_5.amr

.\evaluation\evaluate_bow.ps1 `
    -test .\out\O-Apanhador-no-Campo-de-Centeio\DohareEtAl2018.bow `
    -gold .\out\O-Apanhador-no-Campo-de-Centeio\SumarioExtrativo_1.bow, .\out\O-Apanhador-no-Campo-de-Centeio\SumarioExtrativo_2.bow, .\out\O-Apanhador-no-Campo-de-Centeio\SumarioExtrativo_3.bow, .\out\O-Apanhador-no-Campo-de-Centeio\SumarioExtrativo_4.bow, .\out\O-Apanhador-no-Campo-de-Centeio\SumarioExtrativo_5.bow

Write-Host "...........O-Outro-Lado-da-Meia-Noite...........-" -ForegroundColor red
.\evaluation\evaluate_amr.ps1 `
    -test .\out\O-Outro-Lado-da-Meia-Noite\DohareEtAl2018.amr `
    -gold .\out\O-Outro-Lado-da-Meia-Noite\SumarioExtrativo_1.amr, .\out\O-Outro-Lado-da-Meia-Noite\SumarioExtrativo_2.amr, .\out\O-Outro-Lado-da-Meia-Noite\SumarioExtrativo_3.amr, .\out\O-Outro-Lado-da-Meia-Noite\SumarioExtrativo_4.amr, .\out\O-Outro-Lado-da-Meia-Noite\SumarioExtrativo_5.amr

.\evaluation\evaluate_bow.ps1 `
    -test .\out\O-Outro-Lado-da-Meia-Noite\DohareEtAl2018.bow `
    -gold .\out\O-Outro-Lado-da-Meia-Noite\SumarioExtrativo_1.bow, .\out\O-Outro-Lado-da-Meia-Noite\SumarioExtrativo_2.bow, .\out\O-Outro-Lado-da-Meia-Noite\SumarioExtrativo_3.bow, .\out\O-Outro-Lado-da-Meia-Noite\SumarioExtrativo_4.bow, .\out\O-Outro-Lado-da-Meia-Noite\SumarioExtrativo_5.bow