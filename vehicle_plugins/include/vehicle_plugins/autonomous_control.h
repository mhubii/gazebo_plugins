#ifndef GAZEBO_VEHICLE_PLUGIN_H_
#define GAZEBO_VEHICLE_PLUGIN_H_

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <ignition/math.hh>
#include <stdio.h>
#include <vector>
#include <fstream>
#include <iterator>
#include <ctime>
#include <chrono>

#include <torch/torch.h>
#include <torch/script.h>

#include "keyboard.h"

namespace gazebo
{

class VehiclePlugin : public ModelPlugin
{
public:
	VehiclePlugin();

	void Load(physics::ModelPtr parent, sdf::ElementPtr sdf);	
	
	void OnUpdate();
	void OnCameraMsg(ConstImagesStampedPtr &msg);
	void OnCollisionMsg(ConstContactsPtr &contacts);

	static const uint32_t DOF = 3; // fwd/back, left/right, rotation_left/rotation_right

private:

	// Joint velocity control.
	double vel_[DOF];

	// Reload if goal was hit.
	bool reload_;

	// Initialize joints.
	bool ConfigureJoints(const char* name);

	// Update joints.
	void UpdateJoints();

	// Members.
	physics::ModelPtr model_;

	ignition::math::Pose3d init_pose_;

	event::ConnectionPtr update_connection_;	

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

    std::shared_ptr<torch::jit::script::Module> module_; 
};

// Register the plugin.
GZ_REGISTER_MODEL_PLUGIN(VehiclePlugin)
}

#endif
