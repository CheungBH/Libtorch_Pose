#include <iostream>
#include <utility>
#include <vector>

class Utils {

public:
	//Input the bounding box (xmin,ymin,xmax,ymax) and return the a pair of (x,y) coord of center
	std::pair<double, double> cal_center_point(std::vector<double> box);
	bool is_int_element_in_vector(std::vector<int> v, int element);
	Utils();
	bool is_element_in_vector(std::vector< std::pair<double, double>> v, std::pair<double, double> element);
};