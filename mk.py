"Python utility to re-document/re-build/re-install R packages from the shell"
import os
import subprocess
import argparse
import shutil

parser = argparse.ArgumentParser(
    description=('Mirror src/ directory tree (in markdown) in build/')
)


MATHJAX = ('http://cdn.mathjax.org/mathjax/latest/MathJax.js?'
           'config=TeX-AMS-MML_HTMLorMML')


def build_subdir(arg, dirname, filenames):
    build_dirname = dirname.replace('src', 'build', 1)  # the 1 is needed
    os.makedirs(build_dirname)
    fnames = [fname for fname in filenames if fname.endswith('.md')]
    for fname in fnames:
        output_name = '.'.join([fname[:-len('.md')], 'html'])
        make_pandoc(os.path.join(dirname, fname),
                    os.path.join(build_dirname, output_name))


def make_pandoc(infile, outfile):
    subprocess.call([
        'pandoc',
        '-s',
        '-o', outfile,
        '--mathjax=%s' % MATHJAX,
        '--css=%s' % '/include/solarized-light.css',
        # the syntax highlighting seems to work without setting this, although
        # the numbered blocks are kind of weird (the solarized css is, I think,
        # interfering with pandoc). But it's all usable for now.
        # '--highlight-style', 'zenburn',
        infile
    ])

os.path.walk('src', build_subdir, None)
shutil.copytree('include', 'build/include')
