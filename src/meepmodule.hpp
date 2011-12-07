#include <Python.h>

#include <meep.hpp>

extern "C" {

PyObject *convert_to_object(void *pointer);
int convert_to_pointer(PyObject *object, void *address);
bool component_from_cstring(const char *cstring, meep::component *component);
meep::vec *vector_from_tuple(PyObject *sequence);
    
} // extern "C"