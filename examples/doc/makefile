# Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the MFEM library. For more information and source code
# availability see http://mfem.googlecode.com.
#
# MFEM is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

examples: Examples.conf
	doxygen Examples.conf
	rm -rf ../README_files
	rm -f Examples/html/*.js Examples/html/ftv*
	rm -f Examples/html/graph* Examples/html/nav*
	mv -f Examples/html ../README_files
	rm -rf Examples
	cp images/*.png ../README_files
	cd ..; ln -fs README_files/index.html README.html

clean:
	rm -rf Examples *~

distclean: clean
	rm -rf ../README_files ../README.html

