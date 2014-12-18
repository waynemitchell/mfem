html: MFEM_CodeDocumentation.conf
	doxygen MFEM_CodeDocumentation.conf
	rm -f MFEM_CodeDocumentation.html
	ln -s MFEM_CodeDocumentation_files/html/index.html MFEM_CodeDocumentation.html

clean:
	rm -rf html* MFEM_CodeDocumentation_files* MFEM_CodeDocumentation.html *~
