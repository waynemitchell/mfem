
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
	rm -f $(TESTS) *~
