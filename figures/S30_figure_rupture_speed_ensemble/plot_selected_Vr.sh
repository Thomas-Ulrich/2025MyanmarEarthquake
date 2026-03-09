contour_args="file_index=0 var=RT contour=black,4,0,max,5"
time="i-1"
#file=$1
zoom=1

files=()
indices=(17 156 94 8 93 111 144 5)

for i in "${indices[@]}"; do
    printf -v num "%04d" "$i"
    files+=(extracted_output/dyn_"$num"*fault.xdmf)
done

printf "%s\n" "${files[@]}"
i=0
outputs=""
for f in "${files[@]}"; do
    echo "$f"
   light_quake_visualizer "$f"  --variable Vr_kms           --cmap lapaz_r0  --color_range "0 6"  --time $time --zoom 3.5 --window 2500 600 --output Vr$i #--scalar_bar "0.93 0.35 160"
   outputs="output/Vr$i.png $outputs"
   ((i++))
done


image_combiner --inputs $outputs \
               --rel 0.5 1.0 \
               --output output/Vr_across_ensemble.png \
               --col 1


