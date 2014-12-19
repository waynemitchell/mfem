html: MFEM_CodeDocumentation.conf
	doxygen MFEM_CodeDocumentation.conf
	rm -f MFEM_CodeDocumentation.html
	ln -s MFEM_CodeDocumentation_files/html/index.html MFEM_CodeDocumentation.html

examples: MFEM_ExamplesDocumentation.conf
	doxygen MFEM_ExamplesDocumentation.conf
	cp -r MFEM_ExamplesDocumentation_files ../examples
	ln -fs ../examples/MFEM_ExamplesDocumentation_files/html/index.html ../examples/README.html

clean:
	rm -rf html* MFEM_CodeDocumentation_files* MFEM_CodeDocumentation.html MFEM_ExamplesDocumentation *~

