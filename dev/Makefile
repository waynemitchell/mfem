
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

ALL_FLAGS = $(CXXFLAGS) -I$(MFEM_DIR) $(EIGEN_FLAGS)

TESTS_CPP = $(wildcard *-test.cpp)
TESTS = $(TESTS_CPP:.cpp=)


all: $(TESTS)

%: %.cpp $(MFEM_LIB_FILE)
	$(CXX) $(ALL_FLAGS) $(@).cpp -o $@ $(MFEM_LIBS)

clean:
	rm -f $(TESTS) *~
