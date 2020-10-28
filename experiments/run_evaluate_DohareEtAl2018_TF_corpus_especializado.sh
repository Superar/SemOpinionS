#!/bin/bash

RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${RED}...........Galaxy-SIII...........${NC}"
./evaluation/evaluate_amr.sh \
    ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/Galaxy-SIII/DohareEtAl2018_TF.amr \
    ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/Galaxy-SIII/SumarioExtrativo_1.amr ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/Galaxy-SIII/SumarioExtrativo_2.amr ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/Galaxy-SIII/SumarioExtrativo_3.amr ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/Galaxy-SIII/SumarioExtrativo_4.amr ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/Galaxy-SIII/SumarioExtrativo_5.amr

./evaluation/evaluate_bow.sh \
    ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/Galaxy-SIII/DohareEtAl2018_TF.bow \
    ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/Galaxy-SIII/SumarioExtrativo_1.bow ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/Galaxy-SIII/SumarioExtrativo_2.bow ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/Galaxy-SIII/SumarioExtrativo_3.bow ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/Galaxy-SIII/SumarioExtrativo_4.bow ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/Galaxy-SIII/SumarioExtrativo_5.bow

echo -e "${RED}...........Iphone-5...........${NC}"
./evaluation/evaluate_amr.sh \
    ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/Iphone-5/DohareEtAl2018_TF.amr \
    ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/Iphone-5/SumarioExtrativo_1.amr ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/Iphone-5/SumarioExtrativo_2.amr ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/Iphone-5/SumarioExtrativo_3.amr ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/Iphone-5/SumarioExtrativo_4.amr ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/Iphone-5/SumarioExtrativo_5.amr


./evaluation/evaluate_bow.sh \
    ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/Iphone-5/DohareEtAl2018_TF.bow \
    ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/Iphone-5/SumarioExtrativo_1.bow ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/Iphone-5/SumarioExtrativo_2.bow ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/Iphone-5/SumarioExtrativo_3.bow ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/Iphone-5/SumarioExtrativo_4.bow ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/Iphone-5/SumarioExtrativo_5.bow

echo -e "${RED}...........O-Apanhador-no-Campo-de-Centeio...........${NC}"
./evaluation/evaluate_amr.sh \
    ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/O-Apanhador-no-Campo-de-Centeio/DohareEtAl2018_TF.amr \
    ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/O-Apanhador-no-Campo-de-Centeio/SumarioExtrativo_1.amr ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/O-Apanhador-no-Campo-de-Centeio/SumarioExtrativo_2.amr ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/O-Apanhador-no-Campo-de-Centeio/SumarioExtrativo_3.amr ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/O-Apanhador-no-Campo-de-Centeio/SumarioExtrativo_4.amr ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/O-Apanhador-no-Campo-de-Centeio/SumarioExtrativo_5.amr

./evaluation/evaluate_bow.sh \
    ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/O-Apanhador-no-Campo-de-Centeio/DohareEtAl2018_TF.bow \
    ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/O-Apanhador-no-Campo-de-Centeio/SumarioExtrativo_1.bow ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/O-Apanhador-no-Campo-de-Centeio/SumarioExtrativo_2.bow ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/O-Apanhador-no-Campo-de-Centeio/SumarioExtrativo_3.bow ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/O-Apanhador-no-Campo-de-Centeio/SumarioExtrativo_4.bow ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/O-Apanhador-no-Campo-de-Centeio/SumarioExtrativo_5.bow

echo -e "${RED}...........O-Outro-Lado-da-Meia-Noite...........${NC}"
./evaluation/evaluate_amr.sh \
    ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/O-Outro-Lado-da-Meia-Noite/DohareEtAl2018_TF.amr \
    ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/O-Outro-Lado-da-Meia-Noite/SumarioExtrativo_1.amr ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/O-Outro-Lado-da-Meia-Noite/SumarioExtrativo_2.amr ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/O-Outro-Lado-da-Meia-Noite/SumarioExtrativo_3.amr ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/O-Outro-Lado-da-Meia-Noite/SumarioExtrativo_4.amr ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/O-Outro-Lado-da-Meia-Noite/SumarioExtrativo_5.amr

./evaluation/evaluate_bow.sh \
    ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/O-Outro-Lado-da-Meia-Noite/DohareEtAl2018_TF.bow \
    ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/O-Outro-Lado-da-Meia-Noite/SumarioExtrativo_1.bow ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/O-Outro-Lado-da-Meia-Noite/SumarioExtrativo_2.bow ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/O-Outro-Lado-da-Meia-Noite/SumarioExtrativo_3.bow ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/O-Outro-Lado-da-Meia-Noite/SumarioExtrativo_4.bow ../Resultados/out_DohareEtAl2018_TF_corpus_especializado/O-Outro-Lado-da-Meia-Noite/SumarioExtrativo_5.bow