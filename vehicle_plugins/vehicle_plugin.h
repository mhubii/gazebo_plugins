#ifndef GAZEBO_VEHICLE_PLUGIN_H_
#define GAZEBO_VEHICLE_PLUGIN_H_

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <ignition/math.hh>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <vector>

#include "navigation.h"
#include <torch/torch.h>

namespace gazebo
{

class VehiclePlugin : public ModelPlugin
{
public:
	VehiclePlugin();

	void Load(physics::ModelPtr parent, sdf::ElementPtr /*sdf*/);	
	
	void OnUpdate();
	void OnCameraMsg(ConstImagesStampedPtr &msg);
	void OnCollisionMsg(ConstContactsPtr &contacts);

	static const uint32_t DOF = 3; // fwd/back, left/right, rotation_left/rotation_right

private:

	// Joint velocity control.
	double vel_[DOF];

	// Operating mode.
	enum OperatingMode {
		USER_MANUAL,
		AUTONOMOUSLY
	} op_mode_;

	// Agent.
	bool CreateAgent();
	bool UpdateAgent();

	bool new_state_; // True if new frame need to be processed.

	// Initialize joints.
	bool ConfigureJoints(const char* name);

	// Update joints.
	bool UpdateJoints();

	// Members.
	physics::ModelPtr model;

	event::ConnectionPtr update_connection;	

	std::vector<physics::JointPtr> joints_;

	// Multi camera node and subscriber.
	gazebo::transport::NodePtr multi_camera_node_;
	gazebo::transport::SubscriberPtr multi_camera_sub_;

	// Collision node and subscriber.
	gazebo::transport::NodePtr collision_node_;
	gazebo::transport::SubscriberPtr collision_sub_;

	// Incremental velocity change.
	double vel_delta_;

	// Keyboard for manual operating mode.
	Keyboard* keyboard_;

	//TMP
	torch::Device device_;
	DQN model_;
};

// Register the plugin.
GZ_REGISTER_MODEL_PLUGIN(VehiclePlugin)
}

#endif
