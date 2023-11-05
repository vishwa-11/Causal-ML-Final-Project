#!/bin/sh

for entry in *.tar.gz
do
    # echo "${entry%%.*}"
    mkdir -p "./${entry%%.*}"
    tar -xf "$entry" -C "./${entry%%.*}"
    echo "$entry extracted to ./${entry%%.*}"
done