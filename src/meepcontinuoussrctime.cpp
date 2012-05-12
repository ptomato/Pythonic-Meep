#include <Python.h>

#include <complex>

#include "meepmodule.hpp"

extern "C" {

// Constructor for meep::continuous_src_time
PyObject *
meep_continuous_src_time_new(PyObject *self, PyObject *args, PyObject *kwargs)
{
    Py_complex py_frequency;
    double width = 0.0, start_time = 0.0;
    double end_time = meep::infinity, slowness = 3.0;
    static const char *kwlist[] = {
        "frequency", "width", "start_time", "end_time", "slowness", NULL };

    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "D|dddd", (char **)kwlist,
        &py_frequency, &width, &start_time, &end_time, &slowness))
        return NULL;

    std::complex<double> frequency(py_frequency.real, py_frequency.imag);
    meep::continuous_src_time *instance = new meep::continuous_src_time(
        frequency, width, start_time, end_time, slowness);

    return Py_BuildValue("O&", convert_to_object, instance);
}

// Destructor for meep::continuous_src_time
PyObject *
meep_continuous_src_time_destroy(PyObject *self, PyObject *object)
{
    meep::continuous_src_time *instance;
    convert_to_pointer(object, &instance);
    delete instance;
    Py_RETURN_NONE;
}

// Getter for meep::continuous_src_time::frequency
PyObject *
meep_continuous_src_time_get_frequency(PyObject *self, PyObject *object)
{
    meep::continuous_src_time *instance;
    convert_to_pointer(object, &instance);

    std::complex<double> frequency = instance->frequency();
    Py_complex py_frequency = { real(frequency), imag(frequency) };

    return Py_BuildValue("D", py_frequency);
}

// Setter for meep::continuous_src_time::frequency
PyObject *
meep_continuous_src_time_set_frequency(PyObject *self, PyObject *args)
{
    meep::continuous_src_time *instance;
    Py_complex py_frequency;

    if(!PyArg_ParseTuple(args, "O&D",
        convert_to_pointer, &instance, &py_frequency))
        return NULL;

    std::complex<double> frequency(py_frequency.real, py_frequency.imag);
    instance->set_frequency(frequency);

    Py_RETURN_NONE;
}

} //extern "C"