#!/bin/bash

source activate py36
until python run_webcam.py; do
    echo "'run_webcam.py' crashed with exit code $?. Restarting..." >&2
    sleep 1
done
source deactivate py36