
CXX = g++
CXXFLAGS = -O3

MFEM_DIR = ..
MFEM_LIB_FILE = $(MFEM_DIR)/libmfem.a
MFEM_LIBS = -L$(MFEM_DIR) -lmfem

# If MFEM is compiled to use POSIX clocks:
USE_POSIX_CLOCKS = NO
ifeq ($(USE_POSIX_CLOCKS),YES)
   MFEM_LIBS += -lrt
endif

EIGEN_DIR = ../../eigen
TEST_EIGEN = NO
ifeq ($(TEST_EIGEN),YES)
   EIGEN_FLAGS = -I$(EIGEN_DIR) -DTEST_EIGEN=1
endif

ENABLE_BGQPROF = NO
ifeq ($(ENABLE_BGQPROF),YES)
   # use the bgq back-end mpi compiler:
   CXX = mpixlcxx_r
   # CXXFLAGS = -O2 -qmaxmem=-1 -qnoeh
   CXXFLAGS = -O5 -qmaxmem=-1 -qnoeh
   CXXFLAGS += -DBGQPROF
   MFEM_LIBS += -L/usr/local/tools/mpitrace/lib -lmpihpm -L/bgsys/drivers/ppcfloor/bgpm/lib -lbgpm -lrt -lstdc++ -lhpmprof
endif

ALL_FLAGS = $(CXXFLAGS) -I$(MFEM_DIR) $(EIGEN_FLAGS)

TESTS_CPP = $(wildcard *-test.cpp)
TESTS = $(TESTS_CPP:.cpp=)


all: $(TESTS)

%: %.cpp $(MFEM_LIB_FILE)
	$(CXX) $(ALL_FLAGS) $(@).cpp -o $@ $(MFEM_LIBS)

clean:
	rm -f $(TESTS) *~
