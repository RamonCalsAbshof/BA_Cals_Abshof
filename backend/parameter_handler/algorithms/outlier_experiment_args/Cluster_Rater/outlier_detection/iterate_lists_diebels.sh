#!/bin/bash
set -e

source iteration_params/args[2]_11-17.sh

readonly PARAM_COUNT=15180

function main {
    local counter=0

    while [ "$counter" -lt "$PARAM_COUNT" ]; do
        python3 outlier_experiment_lists.py ${PARAMS[$counter]}
        echo $counter
        counter=$((counter+1))
    done
}

main



