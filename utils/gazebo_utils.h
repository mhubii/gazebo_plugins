#ifndef GAZEBO_UTILS_H_
#define GAZEBO_UTILS_H_

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <Eigen/Core>


#define PI 3.14159265358979323846

std::random_device random_device;
std::mt19937 random_radius(random_device());
std::mt19937 random_angle(random_device());


// Return a random variable of a uniform circular distribution. The center of the 
// distribution is (x,y), and the radius is r.
Eigen::Vector2f UniformCircularRandVar(float x, float y, float r) {

	Eigen::Vector2f rv(0., 0.);

	// Sample uniformly spherical coordinates.
	float sph_r = std::uniform_real_distribution<float>(0., r)(random_radius);
	float sph_a = std::uniform_real_distribution<float>(0., 2*PI)(random_angle);

	// Convert to cartesian space.
	rv(0) = sph_r*sin(sph_a) + x;
	rv(1) = sph_r*cos(sph_a) + y;

	return rv;
}


// Return the distance between the centers of 2 boxes.
inline float CenterDistance(const ignition::math::Box& a, const ignition::math::Box& b) {

	return (a.Center() - b.Center()).Length();
}


// Compute the distance between two bounding boxes.
inline float BoxDistance(const ignition::math::Box& a, const ignition::math::Box& b) {
	
	float sqr_dist = 0;

	if( b.Max().X() < a.Min().X() ) {
		
		float d = b.Max().X() - a.Min().X();
		sqr_dist += d * d;
	}

	else if( b.Min().X() > a.Max().X() ) {
		
		float d = b.Min().X() - a.Max().X();
		sqr_dist += d * d;
	}

	if( b.Max().Y() < a.Min().Y() ) {
		
		float d = b.Max().Y() - a.Min().Y();
		sqr_dist += d * d;
	}

	else if( b.Min().Y() > a.Max().Y() ) {
		
		float d = b.Min().Y() - a.Max().Y();
		sqr_dist += d * d;
	}

	if( b.Max().Z() < a.Min().Z() ) {
		
		float d = b.Max().Z() - a.Min().Z();
		sqr_dist += d * d;
	}
	
	else if( b.Min().Z() > a.Max().Z() ) {
		
		float d = b.Min().Z() - a.Max().Z();
		sqr_dist += d * d;
	}

	return sqrtf(sqr_dist);
}

#endif
