# ConicalSlicer
Generic Non-Planar 3-axis Conical Slicing based on CNC Kitchen's ConicalSlicer for the A1. 
Algorithm is seperated into 3 main steps: Deforming the model upwards based on a cone angle, slicing it in an external slicer, then reforming the gcode back down.
Use any 3D slicer to visualize the gcode before printing.
A1 conical limits depend on the model and nozzle tip. Recommended cone angle limit is 0-15 degrees.
