#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL meep_ARRAY_API
#include <numpy/arrayobject.h>

#include <iostream>
#include <sstream>

#include <meep.hpp>

#include "material.hpp"

extern "C" {

// Convert a C pointer to a Python string object containing the address
PyObject *
convert_to_object(void *pointer)
{
    std::stringstream buffer;
    buffer << std::hex << pointer;
    const char *pointer_string = buffer.str().c_str();
    return Py_BuildValue("s", pointer_string);
}

// Convert a Python string object with a hex address into a C pointer
int
convert_to_pointer(PyObject *object, void *address)
{
    const char *pointer_string = PyString_AsString(object);
    std::stringstream buffer;
    buffer << std::hex << pointer_string;
    buffer >> *((unsigned long *)address);
    return 1; // success
}

// Convert component to meep::component
// component should be guaranteed by the Python module to be a 
// valid lowercase string
bool
component_from_cstring(const char *cstring, meep::component *component)
{
    std::string component_string(cstring);

    if(component_string == "epsilon")
        *component = meep::Dielectric;
    else if(component_string == "ex")
        *component = meep::Ex;
    else if(component_string == "ey")
        *component = meep::Ey;
    else if(component_string == "ez")
        *component = meep::Ez;
    // TODO: other components
    else {
        PyErr_SetString(PyExc_ValueError, "Unsupported component type");
        return false;
    }
    return true;
}

// Convert coordinate tuple (or any Python object supporting the 
// sequence protocol) to meep vector with the proper number of
// dimensions. Returns a heap-allocated meep::vec.
meep::vec *
vector_from_tuple(PyObject *sequence)
{
    // First make sure the coordinate object is a sequence
    PyObject *center_tuple = PySequence_Fast(sequence,
        "Center coordinates must be a tuple or sequence");
    if(!center_tuple)
        return NULL;

    // Convert the sequence to a tuple
    int num_dimensions = PySequence_Length(center_tuple);
    PyObject **coords = PySequence_Fast_ITEMS(center_tuple);
    
    meep::vec *v;
    switch(num_dimensions) {
        case 1:
            v = new meep::vec(PyFloat_AsDouble(coords[0]));
            break;
        case 2:
            v = new meep::vec(PyFloat_AsDouble(coords[0]),
                PyFloat_AsDouble(coords[1]));
            break;
        case 3:
            v = new meep::vec(PyFloat_AsDouble(coords[0]),
                PyFloat_AsDouble(coords[1]),
                PyFloat_AsDouble(coords[2]));
            break;
        default:
            PyErr_SetString(PyExc_ValueError,
                "Coordinate tuple must have 1, 2, or 3 elements");
            Py_DECREF(center_tuple);
            return NULL;
    }
    
    Py_DECREF(center_tuple);
    return v;
}

// why doesn't PyCFunction work here? segfaults
typedef PyObject *(pycfunc)(PyObject *, PyObject *);
extern pycfunc meep_vol2d, meep_grid_volume_destroy,
    meep_continuous_src_time_new, meep_continuous_src_time_destroy,
    meep_continuous_src_time_get_frequency,
    meep_continuous_src_time_set_frequency, meep_pml,
    meep_boundary_region_destroy, meep_structure_new, meep_structure_destroy,
    meep_structure_add_polarizability, meep_fields_new, meep_fields_destroy,
    meep_fields_add_point_source, meep_fields_add_volume_source,
    meep_fields_time, meep_fields_step, meep_fields_output_hdf5,
    meep_fields_get_ndarray, region_material_new, region_material_destroy,
    region_material_add_region, region_material_set_region_epsilon,
    region_material_add_region_polarizability;

// Method table
static PyMethodDef MeepMethods[] = {
    { "vol2d", meep_vol2d, METH_VARARGS },
    { "grid_volume_destroy", meep_grid_volume_destroy, METH_O },
    { "continuous_src_time_new", meep_continuous_src_time_new,
        METH_VARARGS | METH_KEYWORDS },
    { "continuous_src_time_destroy", meep_continuous_src_time_destroy,
        METH_O },
    { "continuous_src_time_get_frequency",
        meep_continuous_src_time_get_frequency, METH_O },
    { "continuous_src_time_set_frequency",
        meep_continuous_src_time_set_frequency, METH_VARARGS },
    { "pml", meep_pml, METH_VARARGS },
    { "boundary_region_destroy", meep_boundary_region_destroy, METH_O },
    { "structure_new", meep_structure_new, METH_VARARGS },
    { "structure_destroy", meep_structure_destroy, METH_O },
    { "structure_add_polarizability", meep_structure_add_polarizability,
        METH_VARARGS },
    { "fields_new", meep_fields_new, METH_O },
    { "fields_destroy", meep_fields_destroy, METH_O },
    { "fields_add_point_source", meep_fields_add_point_source, METH_VARARGS },
    { "fields_add_volume_source", meep_fields_add_volume_source,
        METH_VARARGS },
    { "fields_time", meep_fields_time, METH_O },
    { "fields_step", meep_fields_step, METH_O },
    { "fields_output_hdf5", meep_fields_output_hdf5, METH_VARARGS },
    { "fields_get_ndarray", meep_fields_get_ndarray, METH_VARARGS },
    { "region_material_new", region_material_new, METH_NOARGS },
    { "region_material_destroy", region_material_destroy, METH_O },
    { "region_material_add_region", region_material_add_region, METH_VARARGS },
    { "region_material_set_region_epsilon", region_material_set_region_epsilon,
        METH_VARARGS },
    { "region_material_add_region_polarizability",
    	region_material_add_region_polarizability, METH_VARARGS },
    { NULL, NULL, 0, NULL } // Sentinel
};

// Module initialization function
PyMODINIT_FUNC
init_meep(void)
{
    (void)Py_InitModule("_meep", MeepMethods);
    import_array();
}

} //extern "C"
