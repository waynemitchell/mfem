list(APPEND ALL_EXE_SRCS
  ex1.cpp
  ex2.cpp
  ex3.cpp
  ex4.cpp
  ex5.cpp
  ex6.cpp
  ex7.cpp
  ex8.cpp
  ex9.cpp
  ex10.cpp
  ex14.cpp
  ex15.cpp
  )

if (MFEM_USE_MPI)
  list(APPEND ALL_EXE_SRCS
    ex1p.cpp
    ex2p.cpp
    ex3p.cpp
    ex4p.cpp
    ex5p.cpp
    ex6p.cpp
    ex7p.cpp
    ex8p.cpp
    ex9p.cpp
    ex10p.cpp
    ex11p.cpp
    ex12p.cpp
    ex13p.cpp
    ex14p.cpp
    ex15p.cpp
    )
endif()

# Add one executable per cpp file
add_mfem_examples(ALL_EXE_SRCS)

