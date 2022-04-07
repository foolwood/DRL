#!/usr/bin/env bash
# https://github.com/chaoyuaw/pytorch-coviar

if [ "$#" -ne 2 ]; then
    echo "Usage: ./reencode.sh [input dir] [output dir]"
fi

indir=$1
outdir=$2

mkdir $outdir
if [[ ! -d "${outdir}" ]]; then
  echo "${outdir} doesn't exist. Creating it.";
  mkdir -p ${outdir}
fi

for c in $(ls ${indir})
do
	for inname in $(ls ${indir}/${c}/*avi)
	do
		class_path="$(dirname "$inname")"
		class_name="${class_path##*/}"

		outname="${outdir}/${class_name}/${inname##*/}"
		outname="${outname%.*}.mp4"

		mkdir -p "$(dirname "$outname")"
		ffmpeg -i ${inname} -c:v mpeg4 -f rawvideo ${outname}

	done
done
