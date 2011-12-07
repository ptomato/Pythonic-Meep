#include <vector>
#include <utility>

#include <meep.hpp>

class RegionMaterial : public meep::material_function {

public:
    RegionMaterial() : 
        regions(),
        region_epsilons(),
        polarizability_region_sigmas(),
        polarizabilities(),
        current_polarizability(-1)
    { }
    
    // Overrides
    double chi1p1(meep::field_type ft, const meep::vec &r);
    void sigma_row(meep::component c, double sigrow[3], const meep::vec &r);
    void set_polarizability(meep::field_type ft, double omega, double gamma);

    // Methods
    unsigned add_region(const meep::volume &volume);
    unsigned get_num_regions(void) { return regions.size(); }
    void set_region_epsilon(unsigned region, double eps);
    void add_region_polarizability(unsigned region, double sigma, 
        double omega, double gamma);
    
private:
    std::vector<meep::volume> regions;
    std::vector<double> region_epsilons;
    // Vector (one for each polarizability) of region_sigmas vectors
    std::vector<std::vector<double> > polarizability_region_sigmas;
    std::vector<std::pair<double, double> > polarizabilities;
    int current_polarizability;

    int find_region_for_point(const meep::vec &r);
    int find_polarizability(double omega, double gamma);

}; // class RegionMaterial
