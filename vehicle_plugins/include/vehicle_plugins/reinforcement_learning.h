#ifndef GAZEBO_VEHICLE_PLUGIN_H_
#define GAZEBO_VEHICLE_PLUGIN_H_

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <ignition/math.hh>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <vector>

#include "keyboard.h"
#include "q_learning.h"
#include "models.h"
#include "gazebo_utils.h"
#include <torch/torch.h>

namespace gazebo
{

class VehiclePlugin : public ModelPlugin
{
public:
	VehiclePlugin();

	~VehiclePlugin();

	void Load(physics::ModelPtr parent, sdf::ElementPtr sdf);	
	
	void OnUpdate();
	void OnCameraMsg(ConstImagesStampedPtr &msg);
	void OnCollisionMsg(ConstContactsPtr &contacts);

	static const uint64_t DOF = 3; // fwd/back, left/right, rotation_left/rotation_right

private:

	// Joint velocity control.
	double vel_[DOF];

	// Agent.
	bool CreateAgent();
	bool UpdateAgent();

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

	// Autonomous control.
    bool autonomous_;
	bool new_state_;

	torch::Tensor l_img_; 
    torch::Tensor r_img_;

	// Net.
	torch::Device device_;
	Actor* model_;
};

// Register the plugin.
GZ_REGISTER_MODEL_PLUGIN(VehiclePlugin)
}

#endif
