#!/bin/bash

ls --full-time loss.py model.py

if [ loss.py -nt model.py ]; then
  echo "loss.py is newer than model.py"
fi

if [ model.py -nt loss.py ]; then
  echo "model.py is newer than loss.py"
fi
