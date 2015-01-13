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

html: CodeDocumentation.conf
	doxygen CodeDocumentation.conf
	rm -f CodeDocumentation.html
	ln -s CodeDocumentation/html/index.html CodeDocumentation.html

examples: Examples.conf
	doxygen Examples.conf
	rm -rf ../examples/README_files
	rm -f Examples/html/*.js Examples/html/ftv*
	rm -f Examples/html/graph* Examples/html/nav*
	mv -f Examples/html ../examples/README_files
	rm -rf Examples
	cd ../examples;	ln -fs README_files/index.html README.html

clean:
	rm -rf CodeDocumentation CodeDocumentation.html Examples *~

distclean: clean
	rm -rf ../examples/README_files ../examples/README.html

