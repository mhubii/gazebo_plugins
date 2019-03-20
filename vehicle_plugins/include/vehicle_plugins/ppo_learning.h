#ifndef GAZEBO_VEHICLE_PLUGIN_H_
#define GAZEBO_VEHICLE_PLUGIN_H_

#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ignition/math.hh>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <boost/thread.hpp>

#include "keyboard.h"
#include "models.h"
#include "proximal_policy_optimization.h"
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

	// Initialize joints.
	bool ConfigureJoints(const char* name);

	// Convert a stereo camera image message to a tensor.
	bool MsgToTensor(ConstImagesStampedPtr& msg, torch::Tensor& l_img, torch::Tensor& r_img);

	// Get the reward.
	auto Reward() -> std::tuple<torch::Tensor /*reward*/, torch::Tensor /*done*/>;

	// Update the joints, given the action.
	void UpdateJoints(double* vel);

	// Reset environment on final state.
	void ResetEnvironment();

	// Members.
	physics::ModelPtr model_;

	ignition::math::Pose3d init_pose_;

	event::ConnectionPtr update_connection_;	

	std::vector<physics::JointPtr> joints_;

	// Initial positions.
	ignition::math::Vector3d obs_pos_;
	ignition::math::Vector3d goal_pos_;

	// Optimization parameters.
	uint ppo_steps_;        // number of steps until ppo update
	uint mini_batch_size_;  // mini batches for ppo
	uint ppo_epochs_;       // number of epochs for proximal policy optimization
	uint max_episodes_;     // number of episodes until simulation is shutdown
	uint max_steps_;        // number of steps before simulation is reset
	double beta_;           // exploration factor

	float mean_score_;
	float best_score_;

	float reward_win_;
	float reward_loss_;
	float cost_step_;
	float reward_goal_factor_;

	bool randomness_; // add randomnes to the simulation
	bool track_; // track the path of the vehicle
	bool prior_; // load initial network
	bool train_; // train or test
	gazebo::common::Time time_; // track the time
	float start_time_;
	float end_time_;

	bool initialized_;
	bool new_state_;
	bool state_updated_;
	bool reset_;

	// File to track the trajectory.
	std::ofstream out_file_vehicle_;
	std::ofstream out_file_others_;
	std::ofstream out_file_reward_;

	std::string location_;

	// Node for communication.
	gazebo::transport::NodePtr node_;

	// Multi camera node and subscriber.
	gazebo::transport::SubscriberPtr multi_camera_sub_;

	// Publisher to shutdown the simulation.
	gazebo::transport::PublisherPtr server_pub_;

	// Autonomous control.
	ActorCritic ac_;
	torch::optim::Optimizer* opt_;

	final_state final_state_;

	// States and actions and everything.
	int64_t height_;
	int64_t width_;

	torch::Tensor last_left_state_;
	torch::Tensor last_right_state_;
	torch::Tensor last_action_;
	torch::Tensor last_value_;

	VT left_states_;
	VT right_states_;
	VT actions_;
	VT rewards_;
	VT dones_;

	VT log_probs_;
	VT returns_;
	VT values_;

	uint c_; // counter

	float score_;

	// Different rewards.
	float last_goal_distance_;
	float goal_distance_reward_;
	float hit_;
};

// Register the plugin.
GZ_REGISTER_MODEL_PLUGIN(VehiclePlugin)
}

#endif
