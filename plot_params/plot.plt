#!/usr/bin/gnuplot

datafile="params_class_count.dat"
pngfile="params_distri.png"

set grid
set xtics 2 ('[0,1)' 1, '[1,2)' 2, '[2,4)' 3, '[4,8)' 4, '[8,16)' 5, \
           '[16,32)' 6, '[32,64)' 7, '[64,128)' 8, '[128,' 9) \
           rotate by -45 scale 0.5,0.3

set xlabel "Absolute value" offset 0,1.0
set ylabel "num of parameters" offset 1.0,1.0
set xrange [0:9.5]
set yrange [0:9.5e+6]
set style fill solid border lc rgb "black"
set boxwidth 0.5 relative
set terminal png
set output "params_distri.png"
plot "params_class_count.dat" using 1:2:2 with labels offset 0,0.5 notitle, '' with boxes title "Weight"
