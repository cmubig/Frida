#!/bin/bash
set -x

# runs all images in images/ directory.
# puts per-image results in output/ directory.
# puts aggregate results in webpage/ directory.

indir="$1"
if [[ -z $indir ]]; then
	indir="images"
fi

mkdir -p webpage
mkdir -p applications
mkdir -p output

html="webpage/index.html"

rows="/tmp/rows.html"
rm -f $rows
touch $rows

for p in 5 4 6; do
	for image in $indir/*; do
		base="`basename ${image%.*}`"
		input="${base}.png"
		dir="${base}-${p}"

		if [ -d output/$dir ]; then
			mv output/$dir $dir
		else
			./run_one_image.sh $image $p
		fi

		pattern="$dir/*-final_recursivelevel--fixed_KS-reconstructed.png"
		files=( $pattern )
		output="${files[0]}" 

		rmse="`grep RMSE $dir/log.txt | sed 's/RGB\ RMSE://' | tr -d ' ' | tr '\n' ' '`"

		cat <<EOF >>$rows
<tr> \
  <td><pre><a href="$dir/index.html">$dir</a></pre></td> \
  <td><img width="300" src="$dir/$input"></td> \
  <td><img height="50" style="border:10px solid gray" src="$dir/primary_pigments_color-${p}.png"></td> \
  <td><pre>$rmse</pre></td> \
</tr>
EOF

		mkdir -p webpage/$dir
		mkdir -p applications/$dir
		cp $dir/$input \
			$dir/index.html \
			$dir/log.txt \
			$dir/primary_pigments_color-${p}.png \
			$dir/primary_pigments_*_curve-*.png \
			$dir/Application_Files/*-KM_*.png \
			$dir/Application_Files/*-PD_*.png \
			webpage/$dir
		cp -r $dir/Application_Files applications/$dir
		mv $dir output/
	done
done



cat <<EOF >$html
<html>
<body>
<table width="100%">
EOF

cat $rows | sort >>$html

cat <<EOF >>$html
</table>
</body>
</html>
EOF
