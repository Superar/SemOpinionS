python .\SemOpinionS.py -m DohareEtAl2018 `
    -c ..\Corpora\OpiSums-PT\Textos_AMR\O-Apanhador-no-Campo-de-Centeio\O-Apanhador-no-Campo-de-Centeio.parsed `
    -g ..\Corpora\OpiSums-PT\Sumarios\O-Apanhador-no-Campo-de-Centeio\Extrativos `
    -a ..\Corpora\AMR-PT-OP\AMR-PT-OP-MANUAL\AMR_Aligned.keep `
    -af giza `
    -oie ..\Corpora\OpenIEOut\O-Apanhador-no-Campo-de-Centeio\merged_documents_new.csv `
    --tfidf ..\Corpora\Reviews\b2w-reviews01_ReLi `
    -o ..\Resultados\out_DohareEtAl2018_corpus_generico\O-Apanhador-no-Campo-de-Centeio

python .\SemOpinionS.py -m DohareEtAl2018 `
    -c ..\Corpora\OpiSums-PT\Textos_AMR\Iphone-5\Iphone-5.parsed `
    -g ..\Corpora\OpiSums-PT\Sumarios\Iphone-5\Extrativos `
    -a ..\Corpora\AMR-PT-OP\AMR-PT-OP-MANUAL\AMR_Aligned.keep `
    -af giza `
    -oie ..\Corpora\OpenIEOut\Iphone-5\merged_documents_new.csv `
    --tfidf ..\Corpora\Reviews\b2w-reviews01_ReLi `
    -o ..\Resultados\out_DohareEtAl2018_corpus_generico\Iphone-5

python .\SemOpinionS.py -m DohareEtAl2018 `
    -c ..\Corpora\OpiSums-PT\Textos_AMR\O-Outro-Lado-da-Meia-Noite\O-Outro-Lado-da-Meia-Noite.parsed `
    -g ..\Corpora\OpiSums-PT\Sumarios\O-Outro-Lado-da-Meia-Noite\Extrativos `
    -a ..\Corpora\AMR-PT-OP\AMR-PT-OP-PARSER\AMR_Aligned.keep `
    -af giza `
    -oie ..\Corpora\OpenIEOut\O-Outro-Lado-da-Meia-Noite\merged_documents_new.csv `
    --tfidf ..\Corpora\Reviews\b2w-reviews01_ReLi `
    -o ..\Resultados\out_DohareEtAl2018_corpus_generico\O-Outro-Lado-da-Meia-Noite

python .\SemOpinionS.py -m DohareEtAl2018 `
    -c ..\Corpora\OpiSums-PT\Textos_AMR\Galaxy-SIII\Galaxy-SIII.parsed `
    -g ..\Corpora\OpiSums-PT\Sumarios\Galaxy-SIII\Extrativos `
    -a ..\Corpora\AMR-PT-OP\AMR-PT-OP-PARSER\AMR_Aligned.keep `
    -af giza `
    -oie ..\Corpora\OpenIEOut\Galaxy-SIII\merged_documents_new.csv `
    --tfidf ..\Corpora\Reviews\b2w-reviews01_ReLi `
    -o ..\Resultados\out_DohareEtAl2018_corpus_generico\Galaxy-SIII
