#ifndef GAZEBO_VEHICLE_PLUGIN_H_
#define GAZEBO_VEHICLE_PLUGIN_H_

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <ignition/math.hh>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <vector> 

#include "keyboard.h"
#include "models.h"
#include "q_learning.h"
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

	void Shutdown();

	// Joint velocity control.
	double vel_[DOF];

	// Reload if goal was hit.
	bool reload_;

	// Episodes and steps counters.
	uint n_episodes_;
	uint n_steps_;

	// Agent.
	bool UpdateAgent();

	// Initialize joints.
	bool ConfigureJoints(const char* name);

	// Update joints.
	bool UpdateJoints();

	// Members.
	physics::ModelPtr model_;

	ignition::math::Pose3d init_pose_;

	event::ConnectionPtr update_connection;	

	std::vector<physics::JointPtr> joints_;

	// Node for communication.
	gazebo::transport::NodePtr node_;

	// Multi camera node and subscriber.
	gazebo::transport::SubscriberPtr multi_camera_sub_;

	// Collision node and subscriber.
	gazebo::transport::SubscriberPtr collision_sub_;

	// Publisher to shutdown the simulation.
	gazebo::transport::PublisherPtr server_pub_;

	// Incremental velocity change.
	double vel_delta_;

	// Keyboard for manual operating mode.
	Keyboard* keyboard_;

	// Autonomous control.
	QLearning* brain_;

    bool autonomous_; // on X key
	bool new_state_;

	// States and actions and everything.
	torch::Tensor l_img_;
    torch::Tensor r_img_;

	torch::Tensor action_;
	torch::Tensor reward_;

	// Different rewards.
	float goal_distance_;
	float hit_;

	torch::Tensor l_img_next_;
	torch::Tensor r_img_next_;

	torch::Tensor dones_;
};

// Register the plugin.
GZ_REGISTER_MODEL_PLUGIN(VehiclePlugin)
}

#endif
