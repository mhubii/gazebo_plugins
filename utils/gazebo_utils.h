#ifndef GAZEBO_UTILS_H_
#define GAZEBO_UTILS_H_

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>


// Compute the distance between two bounding boxes.
inline float BoxDistance(const ignition::math::Box& a, const ignition::math::Box& b)
{
	float sqr_dist = 0;

	if( b.Max().X() < a.Min().X() ) {
		
		float d = b.Max().X() - a.Min().X();
		sqr_dist += d * d;
	}

	else if( b.Min().Min() > a.Max().X() ) {
		
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