# Training
python .\SemOpinionS.py -m machine_learning `
    -t ..\Corpora\Training-AMR\all\training `
    -tt ..\Corpora\Training-AMR\all\target `
    -a ..\Corpora\AMR-PT-OP\SPAN-MANUAL\combined_manual_training_target_jamr.txt `
    -af jamr `
    -s ..\Ferramentas\oplexicon_v3.0\lexico_v3.0.txt `
    --tfidf ..\Corpora\Reviews\b2w-reviews01_ReLi `
    -ml mlp `
    -o out

# Run Summarizaztion
python .\SemOpinionS.py -m machine_learning `
    -c ..\Corpora\OpiSums-PT\Textos_AMR\O-Apanhador-no-Campo-de-Centeio\O-Apanhador-no-Campo-de-Centeio.parsed `
    -g ..\Corpora\OpiSums-PT\Sumarios\O-Apanhador-no-Campo-de-Centeio\Extrativos `
    -mo out\model.joblib `
    -a ..\Corpora\AMR-PT-OP\SPAN-MANUAL\combined_manual_training_target_jamr.txt `
    -af jamr `
    -s ..\Ferramentas\oplexicon_v3.0\lexico_v3.0.txt `
    --tfidf ..\Corpora\Reviews\b2w-reviews01_ReLi `
    -oie ..\Corpora\OpenIEOut\O-Apanhador-no-Campo-de-Centeio\merged_documents_new.csv `
    -o out\O-Apanhador-no-Campo-de-Centeio

python .\SemOpinionS.py -m machine_learning `
    -c ..\Corpora\OpiSums-PT\Textos_AMR\Iphone-5\Iphone-5.parsed `
    -g ..\Corpora\OpiSums-PT\Sumarios\Iphone-5\Extrativos `
    -mo out\model.joblib `
    -a ..\Corpora\AMR-PT-OP\SPAN-MANUAL\combined_manual_training_target_jamr.txt `
    -af jamr `
    -s ..\Ferramentas\oplexicon_v3.0\lexico_v3.0.txt `
    --tfidf ..\Corpora\Reviews\b2w-reviews01_ReLi `
    -oie ..\Corpora\OpenIEOut\Iphone-5\merged_documents_new.csv `
    -o out\Iphone-5

python .\SemOpinionS.py -m machine_learning `
    -c ..\Corpora\OpiSums-PT\Textos_AMR\O-Outro-Lado-da-Meia-Noite\O-Outro-Lado-da-Meia-Noite.parsed `
    -g ..\Corpora\OpiSums-PT\Sumarios\O-Outro-Lado-da-Meia-Noite\Extrativos `
    -mo out\model.joblib `
    -a ../Corpora/AMR-PT-OP/SPAN-PARSER/combined_parser_training_target_jamr.txt `
    -af jamr `
    -s ..\Ferramentas\oplexicon_v3.0\lexico_v3.0.txt `
    --tfidf ..\Corpora\Reviews\b2w-reviews01_ReLi `
    -oie ..\Corpora\OpenIEOut\O-Outro-Lado-da-Meia-Noite\merged_documents_new.csv `
    -o out\O-Outro-Lado-da-Meia-Noite

python .\SemOpinionS.py -m machine_learning `
    -c ..\Corpora\OpiSums-PT\Textos_AMR\Galaxy-SIII\Galaxy-SIII.parsed `
    -g ..\Corpora\OpiSums-PT\Sumarios\Galaxy-SIII\Extrativos `
    -mo out\model.joblib `
    -a ../Corpora/AMR-PT-OP/SPAN-PARSER/combined_parser_training_target_jamr.txt `
    -af jamr `
    -s ..\Ferramentas\oplexicon_v3.0\lexico_v3.0.txt `
    --tfidf ..\Corpora\Reviews\b2w-reviews01_ReLi `
    -oie ..\Corpora\OpenIEOut\Galaxy-SIII\merged_documents_new.csv `
    -o out\Galaxy-SIII