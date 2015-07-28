#!/bin/bash
echo "running">test_out.txt
echo "error">test_err.txt
echo "compare mesh">test_mesh_out.txt
echo "compare init">test_init_out.txt
echo "compare sol">test_sol_out.txt
declare -a arr=("-m ../data/periodic-segment.mesh -p 0 -dt 0.005" "-m ../data/periodic-square.mesh -p 0 -dt 0.01" "-m ../data/periodic-hexagon.mesh -p 0 -dt 0.01" "-m ../data/periodic-square.mesh -p 1 -dt 0.005 -tf 9" "-m ../data/periodic-hexagon.mesh -p 1 -dt 0.005 -tf 9" "-m ../data/disc-nurbs.mesh -p 2 -rp 1 -dt 0.005 -tf 9" "-m ../data/periodic-square.mesh -p 3 -rp 2 -dt 0.0025 -tf 9 -vs 20" "-m ../data/periodic-cube.mesh -p 0 -o 2 -rp 1 -dt 0.01 -tf 8")


make clean
cd ../; make -j; cd examples; make ex9p;

for ((j=0; j<=8;j++))
{
for ((i=1; i<=1; i++))
{

for ((N=8; N>=1; N--))
{
  echo "Running with $N processors"
  mpirun -np $N -mca btl ^openib ex9p -s 13 ${arr[$j]} >> test_out.txt 2>> test_err.txt
for ((M=0; M<N; M++))
{
  mv ex9-mesh.00000${M} ex9-mesh-ark.00000${M}
  mv ex9-init.00000${M} ex9-init-ark.00000${M}
  mv ex9-final.00000${M} ex9-final-ark.00000${M}
} 
  mpirun -np $N -mca btl ^openib ex9p -s 4 ${arr[$j]} >> test_out.txt 2>> test_err.txt
  echo $N >>test_mesh_out.txt
  echo $N >>test_init_out.txt
  echo $N >>test_sol_out.txt
for ((M=0; M<N; M++))
{
  diff ex9-mesh.00000${M} ex9-mesh-ark.00000${M} >> test_mesh_out.txt
  diff ex9-init.00000${M} ex9-init-ark.00000${M} >>test_init_out.txt
  diff ex9-final.00000${M} ex9-final-ark.00000${M} >>test_sol_out.txt
} 
}

for ((N=8; N>=1; N--))
{
  echo "Running with $N processors"
  mpirun -np $N -mca btl ^openib ex9p -s 11 >> test_out.txt 2>> test_err.txt
}

}
}

