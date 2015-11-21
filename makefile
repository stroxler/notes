build: clean md2html

serve: build
	cd build; python -m SimpleHTTPServer 8080

clean:
	rm -rf build

md2html:
	python mk.py
