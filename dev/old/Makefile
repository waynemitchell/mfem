
# Use the MFEM build directory
MFEM_DIR = ..
CONFIG_MK = $(MFEM_DIR)/config/config.mk

ifneq (clean,$(MAKECMDGOALS))
   -include $(CONFIG_MK)
endif

EIGEN_DIR = ../../eigen
TEST_EIGEN = NO
ifeq ($(TEST_EIGEN),YES)
   EIGEN_FLAGS = -I$(EIGEN_DIR) -DTEST_EIGEN=1
endif

ENABLE_BGQPROF = NO
ifeq ($(ENABLE_BGQPROF),YES)
   # use the bgq back-end mpi compiler:
   MFEM_CXX = mpixlcxx_r
   # CXXFLAGS = -O2 -qmaxmem=-1 -qnoeh
   MFEM_FLAGS = -O5 -qmaxmem=-1 -qnoeh
   MFEM_FLAGS += -DBGQPROF
   MFEM_LIBS += -L/usr/local/tools/mpitrace/lib -lmpihpm -L/bgsys/drivers/ppcfloor/bgpm/lib -lbgpm -lrt -lstdc++ -lhpmprof
endif

# MFEM_CXXFLAGS = -O3
# MFEM_CXXFLAGS += -march=native
# MFEM_CXXFLAGS += -std=c++11

# Default values in gcc 4.9.2/5.1.0 (from params.def)
#  max-unrolled-insns                     . 200
#  max-average-unrolled-insns             . 80
#  max-unroll-times                       . 8
#  max-peeled-insns                       . 100
#  max-peel-times                         . 16
#  max-peel-branches                      . 32
#  max-completely-peeled-insns            . 100/200
#  max-completely-peel-times              . 16
#  max-once-peeled-insns (new?)           . 400
#  max-completely-peel-loop-nest-depth    . 8

# MFEM_CXXFLAGS += -fpeel-loops\
#  --param max-unrolled-insns=400\
#  --param max-average-unrolled-insns=160\
#  --param max-unroll-times=8\
#  --param max-peeled-insns=200\
#  --param max-peel-times=16\
#  --param max-peel-branches=32\
#  --param max-completely-peeled-insns=200\
#  --param max-completely-peel-times=16\
#  --param max-completely-peel-loop-nest-depth=8

# MFEM_CXXFLAGS += -fopt-info-all=all.info\
#  -fdump-tree-optimized-blocks

TESTS_CPP = $(wildcard *-test.cpp)
TESTS = $(TESTS_CPP:.cpp=)


.PHONY: all clean

all: $(TESTS)

# Remove built-in rule
%: %.cpp

# Replace the default implicit rule for *.cpp files
%: %.cpp $(CONFIG_MK) $(MFEM_LIB_FILE)
	$(MFEM_CXX) $(MFEM_FLAGS) $(EIGEN_FLAGS) $(@).cpp -o $@ $(MFEM_LIBS)

# Generate an error message if the MFEM library is not built and exit
$(CONFIG_MK) $(MFEM_LIB_FILE):
	$(error The MFEM library is not built)

clean:
	rm -rf $(TESTS) *~ *.dSYM
	rm -rf *.info *.optimized
