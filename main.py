from processing_py import *

from graph import Sample

WIN_W = 1000
WIN_H = 1000

app = App(WIN_W, WIN_H)
app.background(255)

source_file = "sources"
pipes_files = "pipes"
sinks_file = "sinks"

spl = Sample(source_file, sinks_file, pipes_files)