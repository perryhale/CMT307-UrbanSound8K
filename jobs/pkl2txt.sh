#!/bin/bash

# validate argument
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_pkl_file>"
    exit 1
fi

# validate path
PKL_FILE="$1"
if [ ! -f "$PKL_FILE" ]; then
    echo "Error: File '$PKL_FILE' not found."
    exit 1
fi

# format output file name by replacing .pkl with .txt
TXT_FILE="${PKL_FILE%.pkl}.txt"

# use Python to load pickle file and print contents, then redirect output to text file
python3 -c "import pickle, sys; print(pickle.load(open(sys.argv[1], 'rb')))" "$PKL_FILE" > "$TXT_FILE"

# trace success message
echo "Saved PKL output to $TXT_FILE"
