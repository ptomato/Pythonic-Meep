#include <Python.h>

#include "meepmodule.hpp"

extern "C" {

// Constructor for meep::structure
PyObject *
meep_structure_new(PyObject *self, PyObject *args, PyObject *kwargs)
{
    meep::grid_volume *volume;
    meep::material_function *material;
    PyObject *pml_list = NULL;
    static const char *kwlist[] = {
        "volume_instance", "material_instance", "boundary_region_list", NULL
    };

    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O&|O", (char **)kwlist,
        convert_to_pointer, &volume,
        convert_to_pointer, &material,
        &pml_list))
        return NULL;

    meep::structure *instance;
    if(pml_list) {
        // First make sure the coordinate object is a sequence
        PyObject *pml_tuple = PySequence_Fast(pml_list,
            "PMLs must be a list or sequence");
        if(!pml_tuple)
            return NULL;

        // Convert the sequence to a tuple
        int num_pmls = PySequence_Length(pml_tuple);
        if(num_pmls == 0) {
            instance = new meep::structure(*volume, *material);
        } else {

            PyObject **py_pmls = PySequence_Fast_ITEMS(pml_tuple);

            meep::boundary_region *pml0;
            convert_to_pointer(PyObject_GetAttrString(py_pmls[0], "_instance"),
                &pml0);

            meep::boundary_region pmls(*pml0);
            int count;
            for(count = 1; count < num_pmls; count++) {
                meep::boundary_region *pml;
                convert_to_pointer(PyObject_GetAttrString(py_pmls[count],
                    "_instance"), &pml);
                pmls = pmls + *pml;
            }

            instance = new meep::structure(*volume, *material, pmls);

        }

        Py_DECREF(pml_tuple);

    } else {
        instance = new meep::structure(*volume, *material);
    }

    return Py_BuildValue("O&", convert_to_object, instance);
}

// Destructor for meep::structure
PyObject *
meep_structure_destroy(PyObject *self, PyObject *object)
{
    meep::structure *instance;
    convert_to_pointer(object, &instance);
    delete instance;
    Py_RETURN_NONE;
}

// Add polarizability resonance
PyObject *
meep_structure_add_polarizability(PyObject *self, PyObject *args)
{
    meep::structure *instance;
    meep::material_function *sigma_instance;
    double omega, gamma;

    if(!PyArg_ParseTuple(args, "O&O&dd",
        convert_to_pointer, &instance,
        convert_to_pointer, &sigma_instance,
        &omega,
        &gamma))
        return NULL;

    instance->add_polarizability(*sigma_instance, omega, gamma);

    Py_RETURN_NONE;
}

} // extern "C"
