#include <Python.h>

#include "meepmodule.hpp"

extern "C" {

// Create a 2D meep::grid_volume
PyObject *
meep_vol2d(PyObject *self, PyObject *args)
{
    double size_x, size_y, resolution;
    if(!PyArg_ParseTuple(args, "(dd)d", &size_x, &size_y, &resolution))
        return NULL;

    meep::grid_volume volume = meep::vol2d(size_x, size_y, resolution);
    // Create a heap-allocated pointer from the stack-allocated grid volume
    meep::grid_volume *heap_volume = new meep::grid_volume(volume);
    heap_volume->center_origin();

    return Py_BuildValue("O&", convert_to_object, heap_volume);
}

// Destructor for meep::grid_volume
PyObject *
meep_grid_volume_destroy(PyObject *self, PyObject *object)
{
    meep::grid_volume *instance;
    convert_to_pointer(object, &instance);
    delete instance;
    Py_RETURN_NONE;
}

} // extern "C"