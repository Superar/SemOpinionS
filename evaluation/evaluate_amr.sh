#!/bin/bash

test=$1
gold=${@:2}

echo "-----------SMATCH-----------"
for g in $gold; do
    echo "$(basename -s .amr $test)-$(basename -s .amr $g)"
    python ./venv/bin/smatch.py -f $test $g
done

echo "------------SEMA------------"
for g in $gold; do
    echo "$(basename -s .amr $test)-$(basename -s .amr $g)"
    python ./venv/sema/sema.py -t $test -g $g
done
