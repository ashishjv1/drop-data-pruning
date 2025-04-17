#!/bin/bash
expect -f - << EOF
spawn rsync -avz --exclude '.git/' --exclude '**/__pycache__/' --exclude '*.pyc' --exclude 'emissions.csv' ./drop_data_pruning/ a.jha@10.5.1.1:/home/a.jha/drop-data-pruning/
expect "password:"
send "thelastgameiplayedin@)19\r"
expect eof
EOF
