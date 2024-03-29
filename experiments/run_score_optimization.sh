# Training
python ./SemOpinionS.py -m score_optimization \
    -t ../Corpora/Training-AMR/all/training \
    -tt ../Corpora/Training-AMR/all/target \
    -a ../Corpora/AMR-PT-OP/SPAN-MANUAL/combined_manual_training_target_jamr.txt \
    -af jamr \
    -s ../Corpora/oplexicon_v3.0/lexico_v3.0.txt \
    --tfidf ../Corpora/Reviews/b2w-reviews01_ReLi \
    -asp ../Corpora/OpiSums-PT/Aspectos/aspects/all.json \
    -o out

# Run Summarizaztion
python ./SemOpinionS.py -m score_optimization \
    -c ../Corpora/OpiSums-PT/Textos_AMR/O-Apanhador-no-Campo-de-Centeio/O-Apanhador-no-Campo-de-Centeio.parsed \
    -g ../Corpora/OpiSums-PT/Sumarios/O-Apanhador-no-Campo-de-Centeio/Extrativos \
    -mo out/weights.csv \
    -a ../Corpora/AMR-PT-OP/SPAN-MANUAL/combined_manual_training_target_jamr.txt \
    -af jamr \
    -s ../Corpora/oplexicon_v3.0/lexico_v3.0.txt \
    --tfidf ../Corpora/Reviews/b2w-reviews01_ReLi \
    -oie ../Corpora/OpenIEOut/O-Apanhador-no-Campo-de-Centeio/merged_documents_new.csv \
    -asp ../Corpora/OpiSums-PT/Aspectos/aspects/ReLi-Salinger.json \
    -o out/O-Apanhador-no-Campo-de-Centeio

python ./SemOpinionS.py -m score_optimization \
    -c ../Corpora/OpiSums-PT/Textos_AMR/Iphone-5/Iphone-5.parsed \
    -g ../Corpora/OpiSums-PT/Sumarios/Iphone-5/Extrativos \
    -mo out/weights.csv \
    -a ../Corpora/AMR-PT-OP/SPAN-MANUAL/combined_manual_training_target_jamr.txt \
    -af jamr \
    -s ../Corpora/oplexicon_v3.0/lexico_v3.0.txt \
    --tfidf ../Corpora/Reviews/b2w-reviews01_ReLi \
    -oie ../Corpora/OpenIEOut/Iphone-5/merged_documents_new.csv \
    -asp ../Corpora/OpiSums-PT/Aspectos/aspects/Iphone-5.json \
    -o out/Iphone-5

python ./SemOpinionS.py -m score_optimization \
    -c ../Corpora/OpiSums-PT/Textos_AMR/O-Outro-Lado-da-Meia-Noite/O-Outro-Lado-da-Meia-Noite.parsed \
    -g ../Corpora/OpiSums-PT/Sumarios/O-Outro-Lado-da-Meia-Noite/Extrativos \
    -mo out/weights.csv \
    -a ../Corpora/AMR-PT-OP/SPAN-PARSER/combined_parser_training_target_jamr.txt \
    -af jamr \
    -s ../Corpora/oplexicon_v3.0/lexico_v3.0.txt \
    --tfidf ../Corpora/Reviews/b2w-reviews01_ReLi \
    -oie ../Corpora/OpenIEOut/O-Outro-Lado-da-Meia-Noite/merged_documents_new.csv \
    -asp ../Corpora/OpiSums-PT/Aspectos/aspects/ReLi-Sheldon-num.json \
    -o out/O-Outro-Lado-da-Meia-Noite

python ./SemOpinionS.py -m score_optimization \
    -c ../Corpora/OpiSums-PT/Textos_AMR/Galaxy-SIII/Galaxy-SIII.parsed \
    -g ../Corpora/OpiSums-PT/Sumarios/Galaxy-SIII/Extrativos \
    -mo out/weights.csv \
    -a ../Corpora/AMR-PT-OP/SPAN-PARSER/combined_parser_training_target_jamr.txt \
    -af jamr \
    -s ../Corpora/oplexicon_v3.0/lexico_v3.0.txt \
    --tfidf ../Corpora/Reviews/b2w-reviews01_ReLi \
    -oie ../Corpora/OpenIEOut/Galaxy-SIII/merged_documents_new.csv \
    -asp ../Corpora/OpiSums-PT/Aspectos/aspects/Galaxy-SIII-num.json \
    -o out/Galaxy-SIII