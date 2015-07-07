#!/bin/bash


./ex9_mfem_rk -m ../data/periodic-segment.mesh -p 0 -r 2 -dt 0.005
./ex9_arkode_rk -m ../data/periodic-segment.mesh -p 0 -r 2 -dt 0.005
echo "1-------------------------------------------------------" > comparison_rk_mfem_arkode.txt
 diff ex9_arkode_rk-final.gf ex9_mfem_rk-final.gf >> comparison_rk_mfem_arkode.txt
./ex9_mfem_rk -m ../data/periodic-square.mesh -p 0 -r 2 -dt 0.01 -tf 10
./ex9_arkode_rk -m ../data/periodic-square.mesh -p 0 -r 2 -dt 0.01 -tf 10
echo "2-------------------------------------------------------" >> comparison_rk_mfem_arkode.txt
 diff ex9_arkode_rk-final.gf ex9_mfem_rk-final.gf >> comparison_rk_mfem_arkode.txt
./ex9_mfem_rk -m ../data/periodic-hexagon.mesh -p 0 -r 2 -dt 0.01 -tf 10
./ex9_arkode_rk -m ../data/periodic-hexagon.mesh -p 0 -r 2 -dt 0.01 -tf 10
echo "3-------------------------------------------------------" >> comparison_rk_mfem_arkode.txt
 diff ex9_arkode_rk-final.gf ex9_mfem_rk-final.gf >> comparison_rk_mfem_arkode.txt
./ex9_mfem_rk -m ../data/periodic-square.mesh -p 1 -r 2 -dt 0.005 -tf 9
./ex9_arkode_rk -m ../data/periodic-square.mesh -p 1 -r 2 -dt 0.005 -tf 9
echo "4-------------------------------------------------------" >> comparison_rk_mfem_arkode.txt
 diff ex9_arkode_rk-final.gf ex9_mfem_rk-final.gf >> comparison_rk_mfem_arkode.txt
./ex9_mfem_rk -m ../data/periodic-hexagon.mesh -p 1 -r 2 -dt 0.005 -tf 9
./ex9_arkode_rk -m ../data/periodic-hexagon.mesh -p 1 -r 2 -dt 0.005 -tf 9
echo "5-------------------------------------------------------" >> comparison_rk_mfem_arkode.txt
 diff ex9_arkode_rk-final.gf ex9_mfem_rk-final.gf >> comparison_rk_mfem_arkode.txt
./ex9_mfem_rk -m ../data/disc-nurbs.mesh -p 2 -r 3 -dt 0.005 -tf 9
./ex9_arkode_rk -m ../data/disc-nurbs.mesh -p 2 -r 3 -dt 0.005 -tf 9
echo "6-------------------------------------------------------" >> comparison_rk_mfem_arkode.txt
 diff ex9_arkode_rk-final.gf ex9_mfem_rk-final.gf >> comparison_rk_mfem_arkode.txt
./ex9_mfem_rk -m ../data/periodic-square.mesh -p 3 -r 4 -dt 0.0025 -tf 9 -vs 20
./ex9_arkode_rk -m ../data/periodic-square.mesh -p 3 -r 4 -dt 0.0025 -tf 9 -vs 20
echo "7-------------------------------------------------------" >> comparison_rk_mfem_arkode.txt
 diff ex9_arkode_rk-final.gf ex9_mfem_rk-final.gf >> comparison_rk_mfem_arkode.txt
./ex9_mfem_rk -m ../data/periodic-cube.mesh -p 0 -r 2 -o 2 -dt 0.02 -tf 8
./ex9_arkode_rk -m ../data/periodic-cube.mesh -p 0 -r 2 -o 2 -dt 0.02 -tf 8
echo "8-------------------------------------------------------" >> comparison_rk_mfem_arkode.txt
 diff ex9_arkode_rk-final.gf ex9_mfem_rk-final.gf >> comparison_rk_mfem_arkode.txt
