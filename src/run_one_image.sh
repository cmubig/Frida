#!/bin/bash
set -x
set -e

if [[ -z `python --version 2>&1 | grep 2.7` ]]; then
	source activate py2_env
fi
if [[ -z `python --version 2>&1 | grep 2.7` ]]; then
	echo "WARNING: not python 2.7, probably won't work!"
fi

input="$1"
pigments="$2"
if [[ -z $pigments ]]; then
	pigments=6
fi

dir="`basename ${input%.*}`-$pigments"
image="`basename ${input%.*}.png`"
imagebase="`basename ${input%.*}`"
log="log.txt"
html="index.html"

prefix="${imagebase}-primary_pigments_color_vertex-${pigments}-KM_weights-W_w_10.0-W_sparse_0.1-W_spatial_1.0-choice_0-blf-W_neighbors_0.0-Recursive_Yes"

if [[ -d $dir ]]; then
	exit 1
fi

mkdir -p $dir
cp wheatfield-crop/Existing* $dir
cp wheatfield-crop/weights-* $dir
convert $input -resize 600x600 $dir/$image
rm -f $dir/$log
rm -f $dir/$html

COUNT=$(expr $pigments - 1)
for i in $(seq 0 $COUNT); do 
	echo $i >> $dir/order1.txt;
done

if true; then
python \
	step1_ANLS_with_autograd.py \
	$image \
	Existing_KS_parameter_KS.txt \
	2 \
	None \
	$imagebase-sampled_pixels-400 \
	0 \
	$pigments \
	10.0 \
	0.0 \
	0.0 \
	0.001 \
	0.001 \
	1e-6 \
	/$dir \
	None \
	0 \
	1 \
	10000 \
	400 \
	1 \
	0 2>&1 | tee -a $dir/$log
fi

if true; then
python \
	fast_energy_RGB_lap_adjusted_weights.py \
	 /$dir \
	 $image \
	 order1.txt \
	 primary_pigments_color_vertex-${pigments}.js \
	 --weights weights-poly3-opaque400-dynamic40000.js \
	 --solve-smaller-factor 2 --save-every 50 2>&1 | tee -a $dir/$log
fi

if true; then
python \
	Extract_PD_palettes.py \
	$image \
	$pigments \
	/$dir 2>&1 | tee -a $dir/$log
fi

cd $dir

if true; then
python \
	../Solve_KM_mixing_model_fixed_KS_with_autograd.py \
	$image \
	primary_pigments_KS-${pigments}.txt  \
	None \
	$prefix \
	10.0 \
	0.1 \
	0 \
	1.0 \
	0.0 \
	blf \
	Yes 2>&1 | tee -a $log
fi

if true; then
python \
	../Solve_KM_layer_model_fixed_KS_with_autograd.py \
	$image \
	primary_pigments_KS-${pigments}.txt  \
	None \
	$prefix-order1 \
	10.0 \
	0.1 \
	0 \
	1.0 \
	0.0 \
	blf \
	Yes \
	order1.txt 2>&1 | tee -a $log
fi

cat <<EOF >>$html
<html>
<body>
<pre>

$prefix

EOF

grep -A5 "RMSE" $log >>$html

cat <<EOF >>$html

</pre>
<h1>Pigments</h1>
<img src="primary_pigments_color-${pigments}.png"><br>
<table width="100%">
<tr><th>K</th>
  <th>S</th>
  <th>K/S</th>
  <th>R</th></tr>
EOF

for i in $(seq $pigments); do
	p=$((i-1))
	cat <<EOF >>$html
<tr><td><img width="100%" src="primary_pigments_K_curve-${p}.png"></td>
  <td><img width="100%" src="primary_pigments_S_curve-${p}.png"></td>
  <td><img width="100%" src="primary_pigments_KS_curve-${p}.png"></td>
  <td><img width="100%" src="primary_pigments_R_curve-${p}.png"></td></tr>
EOF
done

cat <<EOF >>$html
</table>
<h1>Reconstruction</h1>
<table width="100%">
<tr><th>Input</th><th>PD Mixing/Layering Reconstruction</th></tr>
<tr>
<td><img width="100%" src="${image}"></td>
<td><img width="100%" src="${imagebase}-${pigments}-PD_layers-order1-reconstructed.png"</td>
</tr>

<tr><th>KM Mixing Reconstruction</th><th>KM Layering Reconstruction</th></tr>
<tr>
<td><img width="100%" src="${imagebase}-${pigments}-KM_mixing-reconstructed.png"</td>
<td><img width="100%" src="${imagebase}-${pigments}-KM_layers-order1-reconstructed.png"</td>
</tr>
</table>


<h1>KM Mixing Weights</h1>
<img src="primary_pigments_color-${pigments}.png"><br>
<table width="100%">
EOF
for i in $(seq 1 2 $pigments); do
	p=$((i-1))
	q=$((p+1))
	cat <<EOF >>$html
<tr>
<td><img width="100%" src="${imagebase}-${pigments}-KM_mixing-weights_map-0${p}.png"></td>
<td><img width="100%" src="${imagebase}-${pigments}-KM_mixing-weights_map-0${q}.png"></td>
</tr>
EOF
done
cat <<EOF >>$html
</table>


<h1>KM Layering Thickness</h1>
<img src="primary_pigments_color-${pigments}.png"><br>
<table width="100%">
EOF
for i in $(seq 1 2 $pigments); do
	p=$((i-1))
	q=$((p+1))
	cat <<EOF >>$html
<tr>
<td><img width="100%" src="${imagebase}-${pigments}-KM_layers-order1-thickness_map-0${p}.png"></td>
<td><img width="100%" src="${imagebase}-${pigments}-KM_layers-order1-thickness_map-0${q}.png"></td>
</tr>
EOF
done
cat <<EOF >>$html
</table>


<h1>PD Mixing Weights</h1>
<img src="primary_pigments_color-${pigments}.png"><br>
<table width="100%">
EOF
for i in $(seq 1 2 $pigments); do
	p=$((i-1))
	q=$((p+1))
	cat <<EOF >>$html
<tr>
<td><img width="100%" src="${imagebase}-${pigments}-PD_mixing-weights_map-0${p}.png"></td>
<td><img width="100%" src="${imagebase}-${pigments}-PD_mixing-weights_map-0${q}.png"></td>
</tr>
EOF
done
cat <<EOF >>$html
</table>


<h1>PD Layering Opacities</h1>
<img src="primary_pigments_color-${pigments}.png"><br>
<table width="100%">
EOF

for i in $(seq 1 2 $pigments); do
	p=$((i-1))
	q=$((p+1))
	cat <<EOF >>$html
<tr>
<td><img width="100%" src="${imagebase}-${pigments}-PD_layers-order1-opacities_map-0${p}.png"></td>
<td><img width="100%" src="${imagebase}-${pigments}-PD_layers-order1-opacities_map-0${q}.png"></td>
</tr>
EOF
done
cat <<EOF >>$html
</table>


</body>
</html>
EOF
