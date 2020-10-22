#!/bin/bash

test=$1
gold=${@:2}

echo "-----------ROUGE-----------"
for g in $gold; do
    pair="$(basename -s .bow $test)-$(basename -s .bow $g)"
    echo $pair
    python -m rouge_score.rouge \
        --rouge_types=rouge1 \
        --prediction_filepattern=$test \
        --target_filepattern=$g \
        --output_filename="$(dirname $test)/$pair.csv" \
        --noaggregate
done

echo "----------------------------"
