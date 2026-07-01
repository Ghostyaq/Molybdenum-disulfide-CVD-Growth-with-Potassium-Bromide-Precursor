from paraview.simple import *
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np

def get_column(reader, name):
    reader.UpdatePipeline()
    table = servermanager.Fetch(reader)
    return vtk_to_numpy(table.GetColumnByName(name))


reader = CSVReader(
    FileName = ["/Users/mitchellhung/Desktop/Mitchell folder/High School Internships/paraview_data/analysis_results.csv"]
)
cluster = get_column(reader, "cluster")
alpha = np.diff(np.unique(cluster)).min()

table = TableToPoints(
    Input = reader
)

table.XColumn = "x"
table.YColumn = "y"
table.ZColumn = "cluster"

view = GetActiveViewOrCreate("RenderView")

Show(table, view)

surface = Delaunay3D(Input = table)
surface.Alpha = alpha * (2 ** (1/2)) * 0.9
Show(surface, view)

#ColorBy(display, ("POINTS", "cluster"))

Render()