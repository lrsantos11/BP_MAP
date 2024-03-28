using DrWatson
@quickactivate "BP_MAP"

# Here you may include files from the source directory
include(srcdir("BP_MAP.jl"))

println(
"""
Currently active project is: $(projectname())

Path of active project: $(projectdir())


"""
)
