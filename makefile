build: clean md2html

clean:
	rm -rf build

md2html:
	python mk.py
