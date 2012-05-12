#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL meep_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#include <complex>
#include <iostream>

#include "meepmodule.hpp"

extern "C" {

// Constructor for meep::fields
PyObject *
meep_fields_new(PyObject *self, PyObject *object)
{
    meep::structure *structure;
    convert_to_pointer(object, &structure);

    meep::fields *instance = new meep::fields(structure);

    return Py_BuildValue("O&", convert_to_object, instance);
}

// Destructor for meep::fields
PyObject *
meep_fields_destroy(PyObject *self, PyObject *object)
{
    meep::fields *instance;
    convert_to_pointer(object, &instance);
    delete instance;
    Py_RETURN_NONE;
}

// Add a point source to the fields object
PyObject *
meep_fields_add_point_source(PyObject *self, PyObject *args)
{
    meep::fields *fields;
    const char *component_cstring;
    meep::src_time *source;
    PyObject *center_sequence;
    Py_complex py_amplitude = { 1.0, 0.0 };

    if(!PyArg_ParseTuple(args, "O&sO&O|D",
        convert_to_pointer, &fields,
        &component_cstring,
        convert_to_pointer, &source,
        &center_sequence,
        &py_amplitude))
        return NULL;

    meep::component component;
    if(!component_from_cstring(component_cstring, &component))
        return NULL;

    meep::vec *center = vector_from_tuple(center_sequence);
    if(!center)
        return NULL;

    // Convert amplitude to complex<double>
    std::complex<double> amplitude(py_amplitude.real, py_amplitude.imag);

    fields->add_point_source(component, *source, *center, amplitude);

    delete center;
    Py_RETURN_NONE;
}

// Add a volume source to the fields object
PyObject *
meep_fields_add_volume_source(PyObject *self, PyObject *args)
{
    meep::fields *fields;
    const char *component_cstring;
    meep::src_time *source;
    PyObject *point1_seq, *point2_seq;
    Py_complex py_amplitude = { 1.0, 0.0 };

    if(!PyArg_ParseTuple(args, "O&sO&OO|D",
        convert_to_pointer, &fields,
        &component_cstring,
        convert_to_pointer, &source,
        &point1_seq,
        &point2_seq,
        &py_amplitude))
        return NULL;

    meep::component component;
    if(!component_from_cstring(component_cstring, &component))
        return NULL;

    meep::vec *point1 = vector_from_tuple(point1_seq);
    if(!point1)
        return NULL;
    meep::vec *point2 = vector_from_tuple(point2_seq);
    if(!point2) {
        delete point1;
        return NULL;
    }
    meep::volume volume(*point1, *point2);
    delete point1;
    delete point2;

    // Convert amplitude to complex<double>
    std::complex<double> amplitude(py_amplitude.real, py_amplitude.imag);

    fields->add_volume_source(component, *source, volume, amplitude);

    Py_RETURN_NONE;
}

// Get the current simulation time
PyObject *
meep_fields_time(PyObject *self, PyObject *instance)
{
    meep::fields *fields;
    convert_to_pointer(instance, &fields);

    return PyFloat_FromDouble(fields->time());
}

// Run the simulation for one time step
PyObject *
meep_fields_step(PyObject *self, PyObject *instance)
{
    meep::fields *fields;
    convert_to_pointer(instance, &fields);

    fields->step();

    Py_RETURN_NONE;
}

// Output a component to an HDF5 file
PyObject *
meep_fields_output_hdf5(PyObject *self, PyObject *args)
{
    meep::fields *fields;
    meep::grid_volume *volume;
    char *component_str;
    if(!PyArg_ParseTuple(args, "O&sO&",
        convert_to_pointer, &fields,
        &component_str,
        convert_to_pointer, &volume))
        return NULL;

    meep::component component;
    if(!component_from_cstring(component_str, &component))
        return NULL;

    fields->output_hdf5(component, volume->surroundings());

    Py_RETURN_NONE;
}

// Return a field component as a Numpy array
PyObject *
meep_fields_get_ndarray(PyObject *self, PyObject *args)
{
    meep::fields *fields;
    meep::grid_volume *volume;
    char *component_str;
    meep::component component;
    unsigned ndims;
    npy_intp dims[3];
    int x, y, z;
    std::complex<double> *ptr;
    meep::vec coordinate;

    if(!PyArg_ParseTuple(args, "O&O&s",
        convert_to_pointer, &fields,
        convert_to_pointer, &volume,
        &component_str))
        return NULL;

    if(!component_from_cstring(component_str, &component))
        return NULL;

    switch(volume->dim) {
    case meep::D1:
        ndims = 1;
        dims[0] =  volume->nx();
        break;
    case meep::D2:
        ndims = 2;
        dims[0] = volume->nx();
        dims[1] = volume->ny();
        break;
    case meep::D3:
        ndims = 3;
        dims[0] = volume->nx();
        dims[1] = volume->ny();
        dims[2] = volume->nz();
        break;
    default:
        PyErr_SetString(PyExc_ValueError, "Cylindrical dimensions are "
            "not supported yet.");
        return NULL;
    }

    PyObject *ndarray = PyArray_SimpleNew(ndims, dims, NPY_CDOUBLE);
    ptr = (std::complex<double> *)PyArray_DATA(ndarray);

    switch(ndims) {
    case 1:
        coordinate = meep::vec(volume->xmin());
        for(x = 0; x < volume->nx(); x++) {
            *ptr++ = fields->get_field(component, coordinate);
            coordinate += volume->dx();
        }
        break;
    case 2:
        coordinate = meep::vec(volume->xmin(), volume->ymin());
        for(x = 0; x < volume->nx(); x++) {
            for(y = 0; y < volume->ny(); y++) {
                *ptr++ = fields->get_field(component, coordinate);
                coordinate += volume->dy();
            }
            coordinate -= volume->dy() * y;
            coordinate += volume->dx();
        }
        break;
    case 3:
        coordinate = meep::vec(volume->xmin(), volume->ymin(), volume->zmin());
        for(x = 0; x < volume->nx(); x++) {
            for(y = 0; y < volume->ny(); y++) {
                for(z = 0; z < volume->nz(); z++) {
                    *ptr++ = fields->get_field(component, coordinate);
                    coordinate += volume->dz();
                }
                coordinate -= volume->dz() * z;
                coordinate += volume->dy();
            }
            coordinate -= volume->dy() * y;
            coordinate += volume->dx();
        }
        break;
    }

    return ndarray;
}


} // extern "C"
