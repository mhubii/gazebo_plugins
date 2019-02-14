#ifndef GAZEBO_VEHICLE_PLUGIN_H_
#define GAZEBO_VEHICLE_PLUGIN_H_

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <ignition/math.hh>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <boost/thread.hpp>

#include "keyboard.h"
#include "models.h"
#include "q_learning.h"
#include "gazebo_utils.h"
#include <torch/torch.h>

namespace gazebo
{

enum final_state {

	NONE,
	MAXIMUM_STEPS,
	HIT_GOAL,
	HIT_OBSTACLE
};

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

	// Get goal distance.
	float GetGoalDistance();

	// Get obstacle distance.
	float GetObstacleDistance();

	// Print status.
	void PrintStatus();

	// Agent.
	bool UpdateAgent();//state& state);

	// Initialize joints.
	bool ConfigureJoints(const char* name);

	// Get action from brain/keyboard.
	bool GetAction(double* vel);//, state& state);

	// Convert an action to a velocity.
	void ActionToVelocity(torch::Tensor& action, double* vel);

	// Convert a stereo camera image message to a tensor.
	bool MsgToTensor(ConstImagesStampedPtr& msg, torch::Tensor& l_img, torch::Tensor& r_img);

	// Update the joints, given the action.
	void UpdateJoints(double* vel);

	// Reset environment on final state.
	void ResetEnvironment();

	// Members.
	physics::ModelPtr model_;

	ignition::math::Pose3d init_pose_;

	event::ConnectionPtr update_connection;	

	std::vector<physics::JointPtr> joints_;

	// Initial positions.
	ignition::math::Vector3d obs_pos_;
	ignition::math::Vector3d goal_pos_;

	// Optimization parameters.
	uint batch_size_;
	uint buffer_size_;
	uint max_episodes_;
	uint max_steps_;

	float reward_win_;
	float reward_loss_;
	float cost_step_;
	float reward_goal_factor_;

	bool randomness_; // add randomnes to the simulation
	bool track_; // track the path of the vehicle

	// File to track the trajectory.
	std::ofstream out_file_vehicle_;
	std::ofstream out_file_others_;
	std::ofstream out_file_loss_;

	std::string location_;

	// Loss history of an episode.
	std::vector<float> loss_history_;

	float best_loss_;

	// Node for communication.
	gazebo::transport::NodePtr node_;

	// Multi camera node and subscriber.
	gazebo::transport::SubscriberPtr multi_camera_sub_;

	// Publisher to shutdown the simulation.
	gazebo::transport::PublisherPtr server_pub_;

	// Incremental velocity change.
	double vel_delta_;

	// Keyboard for manual operating mode.
	Keyboard* keyboard_;

	// Autonomous control.
	QLearning* brain_;

    bool autonomous_;
	final_state final_state_;

	// States and actions and everything.
	torch::Tensor l_img_;
    torch::Tensor r_img_;

	torch::Tensor action_;
	torch::Tensor reward_;

	float score_;

	// Different rewards.
	float last_goal_distance_;
	float goal_distance_reward_;
	float hit_;

	torch::Tensor l_img_next_;
	torch::Tensor r_img_next_;

	torch::Tensor dones_;
};

// Register the plugin.
GZ_REGISTER_MODEL_PLUGIN(VehiclePlugin)
}

#endif
