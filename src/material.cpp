#include "Python.h"

#include <vector>
#include <utility>
#include <algorithm>
#include <iostream>

#include <meep.hpp>

#include "meepmodule.hpp"
#include "material.hpp"

// Returns -1 if point not in any regions
int RegionMaterial::
find_region_for_point(const meep::vec &r)
{
    int region;
    // Start iterating at the back, because later regions take
    // precedence over earlier regions
    for(region = int(regions.size()) - 1; region >= 0; region--) {
        if(regions[region].contains(r)) {
            break;
        }
    }
    return region;
}

// Returns -1 if polarizability not found
int RegionMaterial::
find_polarizability(double omega, double gamma)
{
    std::vector<std::pair<double, double> >::iterator iter;
    std::pair<double, double> pair_to_search_for = std::make_pair(omega, gamma);

    iter = std::find(polarizabilities.begin(), polarizabilities.end(),
    	pair_to_search_for);

    if(iter == polarizabilities.end())
    	return -1;
    return iter - polarizabilities.begin(); // index of iterator
}

double RegionMaterial::
chi1p1(meep::field_type ft, const meep::vec &r)
{
    // mu
    if(ft == meep::H_stuff)
        return 1.0;

    // epsilon
    int region = find_region_for_point(r);
    if(region == -1)
        return 1.0; // air
    return region_epsilons[(unsigned)region];
}

void RegionMaterial::
sigma_row(meep::component c, double sigrow[3], const meep::vec &r)
{
	if(current_polarizability == -1) {
		std::cerr << "You must call set_polarizability() before sigma_row()"
			<< std::endl;
		return;
	}

    int region = find_region_for_point(r);
    sigrow[0] = sigrow[1] = sigrow[2] = 0.0;
    if(region != -1) {
        sigrow[meep::component_index(c)] =
            polarizability_region_sigmas
            [(unsigned)current_polarizability]
            [(unsigned)region];
    }
}

void RegionMaterial::
set_polarizability(meep::field_type ft, double omega, double gamma)
{
	current_polarizability = find_polarizability(omega, gamma);
}

static void
push_back_zero(std::vector<double>& v)
{
    v.push_back(0.0);
}

// Returns number of new region
unsigned RegionMaterial::
add_region(const meep::volume &volume)
{
    regions.push_back(volume);
    region_epsilons.push_back(1.0);
    std::for_each(polarizability_region_sigmas.begin(),
    	polarizability_region_sigmas.end(),
    	push_back_zero);
    return unsigned(regions.size() - 1);
}

void RegionMaterial::
set_region_epsilon(unsigned region, double eps)
{
    region_epsilons[region] = eps;
}

void RegionMaterial::
add_region_polarizability(unsigned region, double sigma, double omega,
    double gamma)
{
    int index = find_polarizability(omega, gamma);

    if(index == -1) {
    	std::vector<double> region_sigmas;
    	region_sigmas.resize(get_num_regions(), 0.0);
    	region_sigmas[region] = sigma;
        polarizability_region_sigmas.push_back(region_sigmas);

        polarizabilities.push_back(std::make_pair(omega, gamma));

        current_polarizability = polarizabilities.size() - 1;
        return;
    }

    polarizability_region_sigmas[index][region] = sigma;

    // Set the current polarizability as well
    current_polarizability = index;
}

// ----- WRAPPERS -----

extern "C" {

// Constructor for RegionMaterial
PyObject *
region_material_new(PyObject *self)
{
    RegionMaterial *instance = new RegionMaterial();
    return Py_BuildValue("O&", convert_to_object, instance);
}

// Destructor for RegionMaterial
PyObject *
region_material_destroy(PyObject *self, PyObject *instance)
{
    RegionMaterial *material;
    convert_to_pointer(instance, &material);
    delete material;
    Py_RETURN_NONE;
}

// Wrapper for RegionMaterial::add_region()
PyObject *
region_material_add_region(PyObject *self, PyObject *args)
{
    RegionMaterial *material;
    PyObject *point1_seq, *point2_seq;

    if(!PyArg_ParseTuple(args, "O&OO",
        convert_to_pointer, &material,
        &point1_seq,
        &point2_seq))
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

    unsigned region = material->add_region(volume);

    return Py_BuildValue("I", region);
}

// Wrapper for RegionMaterial::set_region_epsilon()
PyObject *
region_material_set_region_epsilon(PyObject *self, PyObject *args)
{
    RegionMaterial *material;
    unsigned region;
    double epsilon;

    if(!PyArg_ParseTuple(args, "O&Id",
        convert_to_pointer, &material,
        &region,
        &epsilon))
        return NULL;

    material->set_region_epsilon(region, epsilon);

    Py_RETURN_NONE;
}

// Wrapper for RegionMaterial::add_region_polarizability()
PyObject *
region_material_add_region_polarizability(PyObject *self, PyObject *args)
{
    RegionMaterial *material;
    unsigned region;
    double sigma, omega, gamma;

    if(!PyArg_ParseTuple(args, "O&Iddd",
        convert_to_pointer, &material,
        &region,
        &sigma,
        &omega,
        &gamma))
        return NULL;

    material->add_region_polarizability(region, sigma, omega, gamma);

    Py_RETURN_NONE;
}

} // extern "C"
