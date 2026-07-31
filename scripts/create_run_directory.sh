#!/bin/bash

# Creates a new run directory, copies all required files,
# renames latest field files to *.restart, and updates startFrom in the .par.
#
#   usage:  ./create_run_directory.sh model_name
#

MODEL=""
for arg in "$@"; do
    case "$arg" in
        -*) echo "Usage: $0 model_name"; exit 1 ;;
        *)  MODEL="$arg" ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Error: model_name required"; exit 1
fi

# Count existing run directories and find the latest
counter=0
latest_tag=""
for f in run[0-9]*; do
    if [ -d "$f" ] && [[ "$f" =~ ^run[0-9]+$ ]]; then
        echo "Found directory with name: $f"
        counter=$((counter+1))
        latest_tag=${f:3}
    fi
done

# Determine source and new directory names
if [[ $counter -eq 0 ]]; then
    srcdir="."
    newdir="run00"
else
    srcdir="run${latest_tag}"
    no_zero=$(echo "$latest_tag" | sed 's/^0*//')
    newtag=$((no_zero+1))
    if [[ $newtag -lt 10 ]]; then
        newdir="run0${newtag}"
    else
        newdir="run${newtag}"
    fi
fi

echo "Source directory: ${srcdir}"
echo "New run directory: ${newdir}"

mkdir "$newdir"

# --- Helper: find latest field file with fallback to *.restart ---
# Usage: find_latest_field <dir> <prefix>   (prefix = MODEL or bfdMODEL)
find_latest_field() {
    local dir="$1"
    local prefix="$2"
    local latest
    latest=$(ls "${dir}/${prefix}0.f"* -rt 2>/dev/null | tail -n1)
    if [[ -n "$latest" ]]; then
        echo "$latest"
    elif [[ -f "${dir}/${prefix}0.restart" ]]; then
        echo "${dir}/${prefix}0.restart"
    fi
}

# Copy static files
cp    "${srcdir}/jobscript.sh"                      "$newdir/"
cp    "${srcdir}/makenek.sh"                        "$newdir/" 2>/dev/null
cp    "${srcdir}/makenek"                           "$newdir/" 2>/dev/null
cp -P "${srcdir}/${MODEL}.usr"                      "$newdir/"
cp    "${srcdir}/${MODEL}.par"                      "$newdir/"
cp    "${srcdir}/${MODEL}.re2"                      "$newdir/"
cp    "${srcdir}/${MODEL}.ma2"                      "$newdir/"
cp    "${srcdir}/element_edges"*                    "$newdir/" 2>/dev/null
cp    "${srcdir}/${MODEL}_periodic_pairs.dat"       "$newdir/" 2>/dev/null
cp    "${srcdir}/SIZE"                              "$newdir/"
cp    "${srcdir}/nek5000"                           "$newdir/"

# Copy velocity/hydro field file (latest numbered or .restart fallback)
fieldfile=$(find_latest_field "$srcdir" "$MODEL")
if [[ -n "$fieldfile" ]]; then
    cp "$fieldfile" "$newdir/${MODEL}0.restart"
    echo "Copied field:     $fieldfile -> $newdir/${MODEL}0.restart"
else
    echo "WARNING: no field file found for ${MODEL}0"
fi

# Update jobscript cd path
sed -i "s|cd .*/run[0-9]*|cd $(pwd)/$newdir|g" "$newdir/jobscript.sh"

# Update startFrom in par to use canonical *.restart names
sed -i "s|^[[:space:]]*startFrom[[:space:]]*=.*|startFrom = ${MODEL}0.restart|" "$newdir/${MODEL}.par"

# Generate SESSION.NAME
echo "$MODEL"          > "$newdir/SESSION.NAME"
echo "$(pwd)/$newdir/" >> "$newdir/SESSION.NAME"

echo "Done: $(pwd)/$newdir"
