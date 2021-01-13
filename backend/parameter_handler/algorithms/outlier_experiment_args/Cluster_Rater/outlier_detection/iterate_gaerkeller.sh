#!/bin/bash
set -e

source iteration_params/args[3]_11-17.sh

readonly PARAM_COUNT=67525
readonly PARALLEL_COUNT=32

function main {
    local counter=777
    local nb_running="\j"
    
    while [ "$counter" -lt "$PARAM_COUNT" ]; do
        while [[ "${nb_running@P}" -lt $PARALLEL_COUNT && $counter -lt "$PARAM_COUNT" ]]; do
            python3 iterate_outlier_experiment.py ${PARAMS[$counter]}&
            echo $counter
            counter=$((counter+1)) 
        done
        
        sleep 0.7m
    done
   
}

main



