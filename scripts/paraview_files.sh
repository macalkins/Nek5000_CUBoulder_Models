#!/bin/bash

MODEL=$1

TAR_FILE="paraview.tar.gz"
DIRECTORY="paraview"

rm -rf "$DIRECTORY"
rm -f "$TAR_FILE"

mkdir -p "$DIRECTORY"
 
visnek $MODEL 
cp $MODEL".nek5000" "$DIRECTORY"
cp $MODEL"0."* "$DIRECTORY"

tar -czvf "$TAR_FILE" "$DIRECTORY"
