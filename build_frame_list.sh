#!/bin/bash

OUTFILE="input_images.txt"
rm -f $OUTFILE

# Folder paths
FOLDER1="src/outputs/run_04_22__21_21_46/50_stroke"
FOLDER2="src/outputs/run_04_22__22_19_16/50_stroke"

# j = 0 to 5
for ((j=0; j<=5; j++)); do
    # First folder: i = 0 to 2340
    for ((i=0; i<=2340; i+=20)); do
        file="$FOLDER1/canvas_04_22__21_21_46_${i}_${j}.png"
        [[ -f "$file" ]] && echo "file '$file'" >> $OUTFILE
    done

    # Second folder: i = 0 to 1720
    for ((i=0; i<=1720; i+=20)); do
        file="$FOLDER2/canvas_04_22__22_19_16_${i}_${j}.png"
        [[ -f "$file" ]] && echo "file '$file'" >> $OUTFILE
    done
done
