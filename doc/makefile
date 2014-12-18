html: MFEM_CodeDocumentation.conf
	doxygen MFEM_CodeDocumentation.conf
	rm -rf MFEM_CodeDocumentation_files
	mv html MFEM_CodeDocumentation_files
	cp index.html MFEM_CodeDocumentation_files
	rm -f MFEM_CodeDocumentation.html
	ln -s MFEM_CodeDocumentation_files/index.html MFEM_CodeDocumentation.html

clean:
	rm -rf html* MFEM_CodeDocumentation_files* MFEM_CodeDocumentation.html *~
