#include <Python.h>

#include "meepmodule.hpp"

extern "C" {

PyObject *
meep_pml(PyObject *self, PyObject *args)
{
    double thickness;
    if(!PyArg_ParseTuple(args, "d", &thickness))
        return NULL;

    meep::boundary_region pml = meep::pml(thickness);
    // Create a heap-allocated pointer from the stack-allocated boundary region
    meep::boundary_region *heap_pml = new meep::boundary_region(pml);
    
    return Py_BuildValue("O&", convert_to_object, heap_pml);
}

// Destructor for meep::boundary_region
PyObject *
meep_boundary_region_destroy(PyObject *self, PyObject *object)
{
    meep::boundary_region *instance;
    convert_to_pointer(object, &instance);
    delete instance;
    Py_RETURN_NONE;
}

} // extern "C"