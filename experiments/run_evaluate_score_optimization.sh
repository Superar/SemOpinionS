#!/bin/bash

RED='\033[0;31m'
NC='\033[0m' # No Color


echo -e "${RED}...........Galaxy-SIII...........${NC}"
./evaluation/evaluate_amr.sh \
    ./out/Galaxy-SIII/score_optimization.amr \
    ./out/Galaxy-SIII/SumarioExtrativo_1.amr ./out/Galaxy-SIII/SumarioExtrativo_2.amr ./out/Galaxy-SIII/SumarioExtrativo_3.amr ./out/Galaxy-SIII/SumarioExtrativo_4.amr ./out/Galaxy-SIII/SumarioExtrativo_5.amr

./evaluation/evaluate_bow.sh \
    ./out/Galaxy-SIII/score_optimization.bow \
    ./out/Galaxy-SIII/SumarioExtrativo_1.bow ./out/Galaxy-SIII/SumarioExtrativo_2.bow ./out/Galaxy-SIII/SumarioExtrativo_3.bow ./out/Galaxy-SIII/SumarioExtrativo_4.bow ./out/Galaxy-SIII/SumarioExtrativo_5.bow

echo -e "${RED}...........Iphone-5...........${NC}"
./evaluation/evaluate_amr.sh \
    ./out/Iphone-5/score_optimization.amr \
    ./out/Iphone-5/SumarioExtrativo_1.amr ./out/Iphone-5/SumarioExtrativo_2.amr ./out/Iphone-5/SumarioExtrativo_3.amr ./out/Iphone-5/SumarioExtrativo_4.amr ./out/Iphone-5/SumarioExtrativo_5.amr


./evaluation/evaluate_bow.sh \
    ./out/Iphone-5/score_optimization.bow \
    ./out/Iphone-5/SumarioExtrativo_1.bow ./out/Iphone-5/SumarioExtrativo_2.bow ./out/Iphone-5/SumarioExtrativo_3.bow ./out/Iphone-5/SumarioExtrativo_4.bow ./out/Iphone-5/SumarioExtrativo_5.bow

echo -e "${RED}...........O-Apanhador-no-Campo-de-Centeio...........${NC}"
./evaluation/evaluate_amr.sh \
    ./out/O-Apanhador-no-Campo-de-Centeio/score_optimization.amr \
    ./out/O-Apanhador-no-Campo-de-Centeio/SumarioExtrativo_1.amr ./out/O-Apanhador-no-Campo-de-Centeio/SumarioExtrativo_2.amr ./out/O-Apanhador-no-Campo-de-Centeio/SumarioExtrativo_3.amr ./out/O-Apanhador-no-Campo-de-Centeio/SumarioExtrativo_4.amr ./out/O-Apanhador-no-Campo-de-Centeio/SumarioExtrativo_5.amr

./evaluation/evaluate_bow.sh \
    ./out/O-Apanhador-no-Campo-de-Centeio/score_optimization.bow \
    ./out/O-Apanhador-no-Campo-de-Centeio/SumarioExtrativo_1.bow ./out/O-Apanhador-no-Campo-de-Centeio/SumarioExtrativo_2.bow ./out/O-Apanhador-no-Campo-de-Centeio/SumarioExtrativo_3.bow ./out/O-Apanhador-no-Campo-de-Centeio/SumarioExtrativo_4.bow ./out/O-Apanhador-no-Campo-de-Centeio/SumarioExtrativo_5.bow

echo -e "${RED}...........O-Outro-Lado-da-Meia-Noite...........${NC}"
./evaluation/evaluate_amr.sh \
    ./out/O-Outro-Lado-da-Meia-Noite/score_optimization.amr \
    ./out/O-Outro-Lado-da-Meia-Noite/SumarioExtrativo_1.amr ./out/O-Outro-Lado-da-Meia-Noite/SumarioExtrativo_2.amr ./out/O-Outro-Lado-da-Meia-Noite/SumarioExtrativo_3.amr ./out/O-Outro-Lado-da-Meia-Noite/SumarioExtrativo_4.amr ./out/O-Outro-Lado-da-Meia-Noite/SumarioExtrativo_5.amr

./evaluation/evaluate_bow.sh \
    ./out/O-Outro-Lado-da-Meia-Noite/score_optimization.bow \
    ./out/O-Outro-Lado-da-Meia-Noite/SumarioExtrativo_1.bow ./out/O-Outro-Lado-da-Meia-Noite/SumarioExtrativo_2.bow ./out/O-Outro-Lado-da-Meia-Noite/SumarioExtrativo_3.bow ./out/O-Outro-Lado-da-Meia-Noite/SumarioExtrativo_4.bow ./out/O-Outro-Lado-da-Meia-Noite/SumarioExtrativo_5.bow
