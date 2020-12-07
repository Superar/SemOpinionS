# Training
python SemOpinionS.py -m LiaoEtAl2018 `
    -a ..\Corpora\AMR-PT-OP\SPAN-MANUAL\combined_manual_training_target_jamr.txt `
    -af jamr `
    -t ..\Corpora\Training-AMR\all\training `
    -tt ..\Corpora\Training-AMR\all\target `
    --tfidf ..\Corpora\Reviews\b2w-reviews01_ReLi `
    -l ramp `
    -sim concept_coverage `
    -o out

# Run summarization
python SemOpinionS.py -m LiaoEtAl2018 `
    -c ..\Corpora\OpiSums-PT\Textos_AMR\O-Apanhador-no-Campo-de-Centeio\O-Apanhador-no-Campo-de-Centeio.parsed `
    -g ..\Corpora\OpiSums-PT\Sumarios\O-Apanhador-no-Campo-de-Centeio\Extrativos `
    -a ..\Corpora\AMR-PT-OP\SPAN-MANUAL\combined_manual_training_target_jamr.txt `
    -af jamr `
    -mo out\weights.csv `
    --tfidf ..\Corpora\Reviews\b2w-reviews01_ReLi `
    -sim concept_coverage `
    -o out\O-Apanhador-no-Campo-de-Centeio 

python SemOpinionS.py -m LiaoEtAl2018 `
    -c ..\Corpora\OpiSums-PT\Textos_AMR\Iphone-5\Iphone-5.parsed `
    -g ..\Corpora\OpiSums-PT\Sumarios\Iphone-5\Extrativos `
    -a ..\Corpora\AMR-PT-OP\SPAN-MANUAL\combined_manual_training_target_jamr.txt `
    -af jamr `
    -mo out\weights.csv `
    --tfidf ..\Corpora\Reviews\b2w-reviews01_ReLi `
    -sim concept_coverage `
    -o out\Iphone-5

python SemOpinionS.py -m LiaoEtAl2018 `
    -c ..\Corpora\OpiSums-PT\Textos_AMR\O-Outro-Lado-da-Meia-Noite\O-Outro-Lado-da-Meia-Noite.parsed `
    -g ..\Corpora\OpiSums-PT\Sumarios\O-Outro-Lado-da-Meia-Noite\Extrativos `
    -a ..\Corpora\AMR-PT-OP\SPAN-PARSER\combined_parser_training_target_jamr.txt `
    -af jamr `
    -mo out\weights.csv `
    --tfidf ..\Corpora\Reviews\b2w-reviews01_ReLi `
    -sim concept_coverage `
    -o out\O-Outro-Lado-da-Meia-Noite

python SemOpinionS.py -m LiaoEtAl2018 `
    -c ..\Corpora\OpiSums-PT\Textos_AMR\Galaxy-SIII\Galaxy-SIII.parsed `
    -g ..\Corpora\OpiSums-PT\Sumarios\Galaxy-SIII\Extrativos `
    -a ..\Corpora\AMR-PT-OP\SPAN-PARSER\combined_parser_training_target_jamr.txt `
    -af jamr `
    -mo out\weights.csv `
    --tfidf ..\Corpora\Reviews\b2w-reviews01_ReLi `
    -sim concept_coverage `
    -o out\Galaxy-SIII
