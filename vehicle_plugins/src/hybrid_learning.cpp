#include "hybrid_learning.h"

#define L_FRONT_PITCH "vehicle::l_front_wheel_pitch"
#define L_FRONT_ROLL "vehicle::l_front_wheel_roll"
#define R_FRONT_PITCH "vehicle::r_front_wheel_pitch"
#define R_FRONT_ROLL "vehicle::r_front_wheel_roll"
#define L_BACK_PITCH "vehicle::l_back_wheel_pitch"
#define L_BACK_ROLL "vehicle::l_back_wheel_roll"
#define R_BACK_PITCH "vehicle::r_back_wheel_pitch"
#define R_BACK_ROLL "vehicle::r_back_wheel_roll"

#define VELOCITY_MIN -5.0f
#define VELOCITY_MAX  5.0f
#define N_ACTIONS 4

#define BATCH_SIZE 128
#define BUFFER_SIZE 2560
#define MAX_EPISODES 100
#define MAXRe_STEPS 600

#define REWARD_WIN  1000
#define REWARD_LOSS -1000
#define COST_STEP 0.01f
#define REWARD_GOAL_FACTOR 500.f

#define WORLD_NAME "vehicle_world"
#define VEHICLE_NAME "vehicle"
#define GOAL_COLLISION "goal::goal::goal_collision"
#define COLLISION_FILTER "ground_plane::link::collision"

namespace gazebo
{

VehiclePlugin::VehiclePlugin() :
	ModelPlugin(), node_(new gazebo::transport::Node()) {

	for (int i = 0; i < DOF; i++) {
	
		vel_[i] = 0.;
	}

	// Initial positions.
	obs_pos_ = ignition::math::Vector3d::Zero;
	goal_pos_ = ignition::math::Vector3d::Zero;

	// Optimization parameters.
	batch_size_ = 128;
	buffer_size_ = 2560;
	max_episodes_ = 100;
	max_steps_ = 600;

	reward_win_ = 1000.;
	reward_loss_ = -1000.;
	cost_step_ = 0.01;
	reward_goal_factor_ = 500.;

	randomness_ = true;
	track_ = false;
	prior_ = false;
	train_ = true;

	start_time_ = 0.;
	end_time_ = 0.;

	new_state_ = true;
	state_updated_ = false;

	best_loss_ = std::numeric_limits<float>::max()/2.;
	best_score_ = 0.;

	reload_ = false;
	n_episodes_ = 0;
	n_steps_ = 0;
	vel_delta_ = 1.0;
	autonomous_ = false;
	final_state_ = NONE;

	// States.
	l_img_ = torch::zeros({}, torch::kUInt8);
	r_img_ = torch::zeros({}, torch::kUInt8);

	// Actions and rewards.
	action_ = torch::zeros({1, 1}, torch::kLong);
	reward_ = torch::zeros({1, 1}, torch::kFloat32);

	score_ = 0.;

	last_goal_distance_ = 0;
	goal_distance_reward_ = 0;
	hit_ = 0;

	// Next states.
	l_img_next_ = torch::zeros({}, torch::kUInt8);
	r_img_next_ = torch::zeros({}, torch::kUInt8);

	dones_ = torch::zeros({1, 1}, torch::kInt);
}

VehiclePlugin::~VehiclePlugin() {

	if (autonomous_) {

		delete brain_;
	}

	if (track_) {

		out_file_vehicle_.close();	
		out_file_others_.close();	
		out_file_loss_.close();
	}
}

void VehiclePlugin::Load(physics::ModelPtr parent, sdf::ElementPtr sdf) {

	// Store the pointer to the model.
	this->model_ = parent;

	// Get the initial positions of the goal and the obstacle.
	obs_pos_ = this->model_->GetWorld()->ModelByName("obstacle")->WorldPose().Pos();
	goal_pos_ = this->model_->GetWorld()->ModelByName("goal")->WorldPose().Pos();

	// Get model parameters.
	sdf::ElementPtr model_sdf = this->model_->GetLink("chassis")->GetSDF();

	int64_t height = 0;
	int64_t width = 0;

	if (model_sdf->HasElement("sensor")) {

		model_sdf = model_sdf->GetElement("sensor");

		while (model_sdf != nullptr) {

			if (model_sdf->HasElement("camera")) {

				height = model_sdf->GetElement("camera")->GetElement("image")->Get<int>("height");
				width = model_sdf->GetElement("camera")->GetElement("image")->Get<int>("width");
				break;
			}

			model_sdf = model_sdf->GetNextElement("sensor");
		}
	}

	// Resize tensors.
	l_img_.resize_({1, height, width, 3});
	r_img_.resize_({1, height, width, 3});
	l_img_next_.resize_({1, height, width, 3});
	r_img_next_.resize_({1, height, width, 3});

	printf("VehicleHybridLearning -- got an input image of size %dx%d.\n", (int)height, (int)width);

	// Get sdf parameters.
	if (sdf->HasElement("autonomous")) {

        autonomous_ = sdf->Get<bool>("autonomous");

		std::string mode = sdf->GetElement("autonomous")->Get<std::string>("mode");

		if (mode.empty()) {

			printf("VehicleHybridLearning -- please provide a mode, either train or test.\n");
		}
		else if (!std::strcmp(mode.c_str(), "test")) {

			printf("VehicleHybridLearning -- running in test mode.\n");
			train_ = false;
		}
		else if (!std::strcmp(mode.c_str(), "train")) {

			printf("VehicleHybridLearning -- running in train mode.\n");
			train_ = true;
		}

		// Optimization parameters.
		batch_size_ = sdf->GetElement("autonomous")->Get<int>("batch_size");
		buffer_size_ = sdf->GetElement("autonomous")->Get<int>("buffer_size");
		max_episodes_  = sdf->GetElement("autonomous")->Get<int>("max_episodes");
		max_steps_ = sdf->GetElement("autonomous")->Get<int>("max_steps");

		reward_win_ = sdf->GetElement("autonomous")->Get<float>("reward_win");
		reward_loss_ = sdf->GetElement("autonomous")->Get<float>("reward_loss");
		cost_step_ = sdf->GetElement("autonomous")->Get<float>("cost_step");
		reward_goal_factor_ = sdf->GetElement("autonomous")->Get<float>("reward_goal_factor");

		randomness_ = sdf->GetElement("autonomous")->Get<bool>("randomness");

		if (autonomous_) {
		
			printf("VehicleHybridLearning -- successfully initialized reinforcement learning in %s mode. \n", mode.c_str());
		}
	}

	if (sdf->HasElement("track")) {

		track_ = sdf->Get<bool>("track");

		location_ = sdf->GetElement("track")->Get<std::string>("location");

		if (track_ && location_.empty()) {

			printf("VehicleHybridLearning -- please provide a location to store the tracked trajectory.");
			track_ = false;
		}
		else if (track_ && !location_.empty()) {

			out_file_vehicle_.open(location_ + "/vehicle_positions.csv");
			out_file_others_.open(location_ + "/goal_obstacle_positions.csv");
			out_file_loss_.open(location_ + "/mean_loss_score.csv");
		}
	}

	// Configure the joints.
	ConfigureJoints(L_FRONT_PITCH);
	ConfigureJoints(R_FRONT_PITCH);
	ConfigureJoints(L_BACK_PITCH);
	ConfigureJoints(R_BACK_PITCH);

	ConfigureJoints(L_FRONT_ROLL);
	ConfigureJoints(R_FRONT_ROLL);
	ConfigureJoints(L_BACK_ROLL);
	ConfigureJoints(R_BACK_ROLL);

	// Store initial pose.
	this->init_pose_ = model_->InitialRelativePose();

	// Create da brain.
	if (autonomous_) {
	
		printf("VehicleHybridLearning -- creating autonomous agent...\n");
		brain_ = new QLearning(3, height, width, N_ACTIONS, batch_size_, buffer_size_, torch::kCUDA);
		printf("VehicleHybridLearning -- successfully initialized agent.\n");
	}

	if (sdf->HasElement("prior")) {

		// Load a prior policy.
		prior_ = sdf->Get<bool>("prior");

		std::string location = sdf->GetElement("prior")->Get<std::string>("location");

		if (prior_ && location.empty()) {

			printf("VehicleHybridLearning -- please provide a location with initial network parameters.\n");
			prior_ = false;
		}
		else if (prior_ && !location.empty()) {

			printf("VehicleHybridLearning -- setting prior network parameters.\n");
			DQN tmp(3, height, width, N_ACTIONS);
			torch::load(tmp, location + "/net.pt");
			brain_->SetPolicy(tmp);
		}
	}

	// Node for communication.
	node_->Init();

	// Create a node for camera communication.
	multi_camera_sub_ = node_->Subscribe("/gazebo/" WORLD_NAME "/" VEHICLE_NAME "/chassis/stereo_camera/images", &VehiclePlugin::OnCameraMsg, this);

	// Create a node for server communication.
	server_pub_ = node_->Advertise<gazebo::msgs::ServerControl>("/gazebo/server/control");

	// Listen to the update event. This event is broadcast every simulation iterartion.
	this->update_connection_ = event::Events::ConnectWorldUpdateBegin(std::bind(&VehiclePlugin::OnUpdate, this));

	// Get current goal distance.
	last_goal_distance_ = GetGoalDistance();

	keyboard_ = Keyboard::Create();

	if (!keyboard_) {

		printf("VehicleHybridLearning -- no keyboard for manual control, shutting down.\n");
		Shutdown();
	}

	start_time_ = time_.Float();
}

void VehiclePlugin::OnUpdate() {

	if (new_state_) { // removed it from OnCameraMsg to this place

		state bundle{torch::zeros({l_img_.size(0), l_img_.size(3), l_img_.size(1), l_img_.size(2)}, torch::kFloat32),
					torch::zeros({r_img_.size(0), r_img_.size(3), r_img_.size(1), r_img_.size(2)}, torch::kFloat32),
					torch::zeros(action_.sizes(), torch::kLong),
					torch::zeros(reward_.sizes(), torch::kFloat32),
					torch::zeros({l_img_next_.size(0), l_img_next_.size(3), l_img_next_.size(1), l_img_next_.size(2)}, torch::kFloat32),
					torch::zeros({r_img_next_.size(0), r_img_next_.size(3), r_img_next_.size(1), r_img_next_.size(2)}, torch::kFloat32),
					torch::zeros(dones_.sizes(), torch::kFloat32)};

		std::get<0>(bundle).copy_(l_img_.to(torch::kFloat32).div(127.5).sub(1.).transpose(1, 3).transpose(2, 3)/*init*/);
		std::get<1>(bundle).copy_(r_img_.to(torch::kFloat32).div(127.5).sub(1.).transpose(1, 3).transpose(2, 3)/*init*/);
		std::get<2>(bundle).copy_(action_.to(torch::kLong)/*action*/);
		std::get<3>(bundle).copy_(reward_.to(torch::kFloat32)/*reward*/);
		std::get<4>(bundle).copy_(l_img_next_.to(torch::kFloat32).div(127.5).sub(1.).transpose(1, 3).transpose(2, 3)/*next*/);
		std::get<5>(bundle).copy_(r_img_next_.to(torch::kFloat32).div(127.5).sub(1.).transpose(1, 3).transpose(2, 3)/*next*/);
		std::get<6>(bundle).copy_(dones_.to(torch::kFloat32)/*dones*/);

		// Update the agent.
		brain_->Step(bundle);

		// Record the loss.
		loss_history_.push_back(brain_->GetLoss());

		if (*(dones_.data<int>()) == 1) {

			ResetEnvironment();
		}
		else {

			// Perform an action.
			action_ = brain_->Act(l_img_next_.to(torch::kFloat32).div(127.5).sub(1.).transpose(1, 3).transpose(2, 3), 
								r_img_next_.to(torch::kFloat32).div(127.5).sub(1.).transpose(1, 3).transpose(2, 3), train_);
			// action_ = brain_->Act(l_img_next_.to(torch::kFloat32).transpose(1, 3).transpose(2, 3) - l_img_.to(torch::kFloat32).transpose(1, 3).transpose(2, 3), 
			// 					  r_img_next_.to(torch::kFloat32).transpose(1, 3).transpose(2, 3) - r_img_.to(torch::kFloat32).transpose(1, 3).transpose(2, 3), train_);

			ActionToVelocity(action_, vel_);
			// GetAction(vel_);

			UpdateJoints(vel_);

			n_steps_ += 1;

			if (max_steps_ <= n_steps_) {

				final_state_ = MAXIMUM_STEPS;
			}

			// Save next image as current image.
			l_img_ = l_img_next_;
			r_img_ = r_img_next_;
		}

		new_state_ = false;
		state_updated_ = true;
	}
}

void VehiclePlugin::OnCameraMsg(ConstImagesStampedPtr &msg) {

	if (state_updated_) {

		if (track_) {

			// Track the position of the vehicle.
			ignition::math::Vector3d pos_vehicle = this->model_->GetWorld()->ModelByName("vehicle")->WorldPose().Pos();
			ignition::math::Vector3d pos_obstacle = this->model_->GetWorld()->ModelByName("obstacle")->WorldPose().Pos();
			ignition::math::Vector3d pos_goal = this->model_->GetWorld()->ModelByName("goal")->WorldPose().Pos();

			out_file_vehicle_ << pos_vehicle[0]  << ", " << pos_vehicle[1]  << ", " << pos_vehicle[2]  << "\n";
		}

		if (!MsgToTensor(msg, l_img_next_, r_img_next_)) {

			printf("VehicleHybridLearning -- could not convert message to tensor.\n");
		};

		// Determine the reward.
		float goal_distance = GetGoalDistance();
		goal_distance_reward_ = last_goal_distance_ - goal_distance;
		last_goal_distance_ = goal_distance;

		reward_[0][0] = reward_goal_factor_*goal_distance_reward_ - cost_step_*n_steps_;

		if (goal_distance < 0.9) {

			final_state_ = HIT_GOAL;
		}
		else if (GetObstacleDistance() < 0.55) {

			final_state_ = HIT_OBSTACLE;
		}

		switch (final_state_)
		{
			case NONE:
				dones_[0][0] = 0;
				break;

			case MAXIMUM_STEPS:
				dones_[0][0] = 1;
				printf("VehicleHybridLearning -- maximum steps reached.\n");
				break;

			case HIT_GOAL:
				dones_[0][0] = 1;
				reward_[0][0] += reward_win_;
				printf("VehicleHybridLearning -- hit goal.\n");
				break;

			case HIT_OBSTACLE:
				dones_[0][0] = 1;
				reward_[0][0] += reward_loss_;
				printf("VehicleHybridLearning -- hit obstacle.\n");
				break;
		
			default:
				printf("VehicleHybridLearning -- vehicle in unknown state.\n");
				break;
		}

		score_ += *(reward_.data<float>());

		// if (n_steps_ % 100 == 0) {

			PrintStatus();
		// }

		state_updated_ = false;
		new_state_ = true;
	}
}

void VehiclePlugin::OnCollisionMsg(ConstContactsPtr &contacts) {

	for (unsigned int i = 0; i < contacts->contact_size(); ++i)
	{
		if( strcmp(contacts->contact(i).collision2().c_str(), COLLISION_FILTER) == 0 ){

			continue;
		}

		std::cout << "Collision between[" << contacts->contact(i).collision1()
				<< "] and [" << contacts->contact(i).collision2() << "]\n";

		final_state_ = (contacts->contact(i).collision1().compare(GOAL_COLLISION) == 0||
						contacts->contact(i).collision2().compare(GOAL_COLLISION) == 0) ? HIT_GOAL : HIT_OBSTACLE;
	}
}

float VehiclePlugin::GetGoalDistance() {

	// Get the goal distance corresponding to the current state.
	physics::ModelPtr goal = this->model_->GetWorld()->ModelByName("goal");
	
	ignition::math::Box goal_box = goal->BoundingBox();
	ignition::math::Box vehicle_box = this->model_->BoundingBox();

	return CenterDistance(vehicle_box, goal_box);
	//return BoxDistance(vehicle_box, goal_box);
}

float VehiclePlugin::GetObstacleDistance() {

	// Get the goal distance corresponding to the current state.
	physics::ModelPtr obstacle = this->model_->GetWorld()->ModelByName("obstacle");
	
	ignition::math::Box obstacle_box = obstacle->BoundingBox();
	ignition::math::Box vehicle_box = this->model_->BoundingBox();

	return CenterDistance(vehicle_box, obstacle_box);
	//return BoxDistance(vehicle_box, obstacle_box);
}

void VehiclePlugin::PrintStatus() {

	// Prints the current status to the command line.
	std::cout << "episode: "       << n_episodes_
			  << "    step: "      << n_steps_ 
			  << "    action: "    << *(action_.to(torch::kCPU).data<long>())
			  << "    dones: "     << *(dones_.data<int>())
			  << "    reward: "    << *(reward_.data<float>())
			  << "    score: "     << score_ 
			  << "    best loss: " << best_loss_ << std::endl;
}

bool VehiclePlugin::UpdateAgent(){//state& some) {

	// Set terminate state.
	dones_[0][0] = reload_ ? 1 : 0;

	// Set reward of performed action.
	reward_[0][0] = reward_goal_factor_*goal_distance_reward_ + hit_ - n_steps_*cost_step_;
	
	if (n_steps_ % 100 == 0) {
			
		PrintStatus();
	}

	// Put states, actions, rewards, and everything together.
	state bundle{torch::empty({l_img_.size(0), l_img_.size(3), l_img_.size(1), l_img_.size(2)}, torch::kFloat32),
				 torch::empty({r_img_.size(0), r_img_.size(3), r_img_.size(1), r_img_.size(2)}, torch::kFloat32),
				 torch::empty(action_.sizes(), torch::kLong),
				 torch::empty(reward_.sizes(), torch::kFloat32),
				 torch::empty({l_img_next_.size(0), l_img_next_.size(3), l_img_next_.size(1), l_img_next_.size(2)}, torch::kFloat32),
				 torch::empty({r_img_next_.size(0), r_img_next_.size(3), r_img_next_.size(1), r_img_next_.size(2)}, torch::kFloat32),
				 torch::empty(dones_.sizes(), torch::kFloat32)};

	std::get<0>(bundle).copy_(l_img_.to(torch::kFloat32).div(127.5).sub(1.).transpose(1, 3).transpose(2, 3)/*init*/);
	std::get<1>(bundle).copy_(r_img_.to(torch::kFloat32).div(127.5).sub(1.).transpose(1, 3).transpose(2, 3)/*init*/);
	std::get<2>(bundle).copy_(action_.to(torch::kLong)/*action*/);
	std::get<3>(bundle).copy_(reward_.to(torch::kFloat32)/*reward*/);
	std::get<4>(bundle).copy_(l_img_next_.to(torch::kFloat32).div(127.5).sub(1.).transpose(1, 3).transpose(2, 3)/*next*/);
	std::get<5>(bundle).copy_(r_img_next_.to(torch::kFloat32).div(127.5).sub(1.).transpose(1, 3).transpose(2, 3)/*next*/);
	std::get<6>(bundle).copy_(dones_.to(torch::kFloat32)/*dones*/);

	brain_->Step(bundle);

	// Perform an action.
	action_ = brain_->Act(l_img_next_.to(torch::kFloat32).div(127.5).sub(1.).transpose(1, 3).transpose(2, 3),
	                      r_img_next_.to(torch::kFloat32).div(127.5).sub(1.).transpose(1, 3).transpose(2, 3), true);

	if (action_.numel() == 0) {

		return false;
	}

 	l_img_ = l_img_next_;
	r_img_ = r_img_next_;

	return true;
}

bool VehiclePlugin::ConfigureJoints(const char* name) {

	std::vector<physics::JointPtr> joints = model_->GetJoints();
	const size_t num_joints = joints.size();

	for (int i = 0; i < num_joints; i++) {

		if (strcmp(name, joints[i]->GetScopedName().c_str()) == 0) {
			
			joints[i]->SetVelocity(0, 0);
			joints_.push_back(joints[i]);
			return true;
		}
	}

	printf("VehicleHybridLearning -- failed to find joint '%s'\n", name);
	return false;
}

bool VehiclePlugin::GetAction(double* vel){//, state& state) {

	keyboard_->Poll();

	if (autonomous_) {

		// printf("new state available: %s\n", (new_state_ ? "true" : "false"));


		// No new processed state.
		// new_state_ = false;

		// State updated flag.
		// state_updated_ = UpdateAgent();//state);

		if (!UpdateAgent()) {
			printf("returning false\n");
			return false;
		}

		// Now done in 
		ActionToVelocity(action_, vel);
		// if (*(action_.to(torch::kCPU).data<long>()) == 0) { // W, action 0 
			
		// 	vel[0] = + vel_delta_;
		// 	vel[1] = + vel_delta_;
		// 	vel[2] = 0;
		// }
		// if (*(action_.to(torch::kCPU).data<long>()) ==  1) { // S, action 1
			
		// 	vel[0] = - vel_delta_;
		// 	vel[1] = - vel_delta_;
		// 	vel[2] = 0;
		// }
		// // if (*(action_.to(torch::kCPU).data<long>()) == 2) { // D, action 2
			
		// // 	vel[0] += vel_delta_;
		// // 	vel[1] -= vel_delta_;
		// // }
		// // if (*(action_.to(torch::kCPU).data<long>()) == 3) { // A, action 3
			
		// // 	vel[0] -= vel_delta_;
		// // 	vel[1] += vel_delta_;
		// // }
		// if (*(action_.to(torch::kCPU).data<long>()) == 2) { // Left, action 4
			
		// 	vel[2] = - vel_delta_;
		// }
		// if (*(action_.to(torch::kCPU).data<long>()) == 3) { // Right, action 5
			
		// 	vel[2] = + vel_delta_;
		// }
		// // if (*(action_.to(torch::kCPU).data<long>()) == 5) { // E, action 6
			
		// // 	for (int i = 0; i < DOF; i++) {
	
		// // 		vel[i] = 0.;
		// // 	}
		// // }
	}

	else if (!autonomous_) {

		if (keyboard_->KeyDown(KEY_W)) {
			
			vel[0] = + vel_delta_;
			vel[1] = + vel_delta_;
		}
		if (keyboard_->KeyDown(KEY_S)) {
			
			vel[0] = - vel_delta_;
			vel[1] = - vel_delta_;
		}
		if (keyboard_->KeyDown(KEY_D)) {
			
			vel[0] = + vel_delta_;
			vel[1] = - vel_delta_;
		}
		if (keyboard_->KeyDown(KEY_A)) {
			
			vel[0] = - vel_delta_;
			vel[1] = + vel_delta_;
		}
		if (keyboard_->KeyDown(KEY_LEFT)) {
			
			vel[2] = - vel_delta_;
		}
		if (keyboard_->KeyDown(KEY_RIGHT)) {
			
			vel[2] = + vel_delta_;
		}
		if (keyboard_->KeyDown(KEY_E)) {
			
			for (int i = 0; i < DOF; i++) {
	
				vel[i] = 0.;
			}
		}
		if (keyboard_->KeyDown(KEY_Q)) {

			printf("VehicleManualControl -- interruption after key q was pressed, shutting down.\n");	
			Shutdown();
		}
	}

	return true;
}

void VehiclePlugin::ActionToVelocity(torch::Tensor& action, double* vel) {

	// Converts the action to a velocity.
	if (*(action.to(torch::kCPU).data<long>()) == 0) { // W, action 0
		
		vel[0] = + vel_delta_;
		vel[1] = + vel_delta_;
		vel[2] = 0.;
	}
	if (*(action.to(torch::kCPU).data<long>()) ==  1) { // S, action 1
		
		vel[0] = - vel_delta_;
		vel[1] = - vel_delta_;
		vel[2] = 0.;
	}
	if (*(action.to(torch::kCPU).data<long>()) == 2) { // D, action 2
		
		vel[0] = + vel_delta_;
		vel[1] = - vel_delta_;
		vel[2] = 0.;
	}
	if (*(action.to(torch::kCPU).data<long>()) == 3) { // A, action 3
		
		vel[0] = - vel_delta_;
		vel[1] = + vel_delta_;
		vel[2] = 0.;
	}
	// if (*(action.to(torch::kCPU).data<long>()) == 4) { // Left, action 4
				
	// 	vel[0] = 0.;
	// 	vel[1] = 0.;
	// 	vel[2] = - vel_delta_;
	// }
	// if (*(action.to(torch::kCPU).data<long>()) == 5) { // Right, action 5
		
	// 	vel[0] = 0.;
	// 	vel[1] = 0.;
	// 	vel[2] = + vel_delta_;
	// }
	// if (*(action.to(torch::kCPU).data<long>()) == 5) { // E, action 6
		
	// 	for (int i = 0; i < DOF; i++) {

	// 		vel[i] = 0.;
	// 	}
	// }
}

bool VehiclePlugin::MsgToTensor(ConstImagesStampedPtr& msg, torch::Tensor& l_img, torch::Tensor& r_img) {

	if (!msg) {

		printf("VehicleHybridLearning -- received NULL message.\n");
		return false;
	}

	const int l_width = msg->image().Get(0).width();
	const int l_height = msg->image().Get(0).height();
	const int l_size = msg->image().Get(0).data().size();
	const int l_bpp = (msg->image().Get(0).step()/msg->image().Get(0).width())*8; // Bits per pixel.

	if (l_bpp != 24) {

		printf("VehicleAReinforcementLearning -- expected 24 bits per pixel uchar3 image from camera, got %i.\n", l_bpp);
		return false;
	}

	if (l_img.sizes() != torch::IntArrayRef({1, l_height, l_width, 3})) {

		printf("VehicleAReinforcementLearning -- resizing tensor to %ix%ix%ix%i", 1, l_height, l_width, 3);
		l_img.resize_({1, l_height, l_width, 3});
	}

	const int r_width = msg->image().Get(1).width();
	const int r_height = msg->image().Get(1).height();
	const int r_size = msg->image().Get(1).data().size();
	const int r_bpp = (msg->image().Get(1).step()/msg->image().Get(0).width())*8; // Bits per pixel.

	if (r_bpp != 24) {

		printf("VehicleAReinforcementLearning -- expected 24 bits per pixel uchar3 image from camera, got %i.\n", r_bpp);
		return false;
	}

	if (r_img.sizes() != torch::IntArrayRef({1, r_height, r_width, 3})) {

		printf("VehicleAReinforcementLearning -- resizing tensor to %ix%ix%ix%i", 1, l_height, l_width, 3);
		r_img.resize_({1, r_height, r_width, 3});
	}

	// Copy image to tensor.
	std::memcpy(l_img.data_ptr(), msg->image().Get(0).data().c_str(), l_size);
	std::memcpy(r_img.data_ptr(), msg->image().Get(1).data().c_str(), r_size);

	return true;
}

void VehiclePlugin::UpdateJoints(double* vel) {

	// Perform an action, given the new velocity.
	// Drive forward/backward and turn.
	joints_[0]->SetVelocity(0, vel[0]); // left
	joints_[1]->SetVelocity(0, vel[1]); // right
	joints_[2]->SetVelocity(0, vel[0]); // left
	joints_[3]->SetVelocity(0, vel[1]); // right

	// Drive left/right. Rotate to frames.
	ignition::math::Vector3<double> axis = axis.UnitX;
	ignition::math::Vector3<double> tmp = tmp.Zero;

	ignition::math::Quaterniond ori = joints_[0]->AxisFrameOffset(0);
	tmp = ori.RotateVector(axis);
	joints_[4]->SetAxis(0, tmp);
	joints_[4]->SetVelocity(0, vel[2]);

	ori = joints_[1]->AxisFrameOffset(0);		
	tmp = ori.RotateVector(axis);
	joints_[5]->SetAxis(0, tmp);
	joints_[5]->SetVelocity(0, vel[2]);

	ori = joints_[2]->AxisFrameOffset(0);
	tmp = ori.RotateVector(axis);
	joints_[6]->SetAxis(0, tmp);
	joints_[6]->SetVelocity(0, vel[2]);

	ori = joints_[3]->AxisFrameOffset(0);
	tmp = ori.RotateVector(axis);
	joints_[7]->SetAxis(0, tmp);
	joints_[7]->SetVelocity(0, vel[2]);
}

void VehiclePlugin::ResetEnvironment() {

	// Analyze the mean loss of this episode.
	float mean_loss = std::accumulate(loss_history_.begin(), loss_history_.end(), 0.)/loss_history_.size();
	float mean_score = score_/n_steps_;

	if (track_) {

		// Track the position of the goal and the obstacle.
		ignition::math::Vector3d pos_obstacle = this->model_->GetWorld()->ModelByName("obstacle")->WorldPose().Pos();
		ignition::math::Vector3d pos_goal = this->model_->GetWorld()->ModelByName("goal")->WorldPose().Pos();

		out_file_others_ << pos_goal[0]     << ", " << pos_goal[1]     << ", " << pos_goal[2]     << ", "
				         << pos_obstacle[0] << ", " << pos_obstacle[1] << ", " << pos_obstacle[2] << "\n";

		end_time_ = time_.Float();

		// Track the mean loss and more.
		out_file_loss_ << n_episodes_ << ", " << mean_loss << ", " << mean_score << ", " << final_state_ << ", " << (end_time_ - start_time_) << "\n";

		start_time_ = end_time_;

		// Save neural net on best mean loss.
		if (mean_score > best_score_) {

			torch::save(brain_->GetTarget(), location_ + "/net.pt");
			best_score_ = mean_score;
		}
	}

	loss_history_.clear();

	n_episodes_ += 1;
	n_steps_ = 0;
	final_state_ = NONE;

	reward_[0][0] = 0.;
	dones_[0][0] = 0;

	score_ = 0.;

	if (max_episodes_ <= n_episodes_) {

		// Shutdown simulation.
		printf("VehicleHybridLearning -- maximal episodes reached, shutting down.\n");
		Shutdown();
	}

	// End episode.
	printf("VehicleHybridLearning -- resetting agent.\n");

	// Reset environment.
	for (int i = 0; i < DOF; i++) {

		vel_[i] = 0.;
	}

	UpdateJoints(vel_);

	model_->Reset();
	model_->ResetPhysicsStates();

	// Reset goal and obstacle to a random state.
	if (randomness_) {

		// Set random obstacle position.
		Eigen::Vector2f obs_rand = UniformCircularRandVar(obs_pos_[0], obs_pos_[1], 0.5);
		ignition::math::Pose3d obs_pose(ignition::math::Vector3d(obs_rand(0), obs_rand(1), 0.), ignition::math::Quaterniond::Identity);
		this->model_->GetWorld()->ModelByName("obstacle")->SetWorldPose(obs_pose);
	
		// Set random goal position.
		Eigen::Vector2f goal_rand = UniformCircularRandVar(goal_pos_[0], goal_pos_[1], 1.);
		ignition::math::Pose3d goal_pose(ignition::math::Vector3d(goal_rand(0), goal_rand(1), 0.5), ignition::math::Quaterniond::Identity);
		this->model_->GetWorld()->ModelByName("goal")->SetWorldPose(goal_pose);
	}

	// Get current goal distance.
	last_goal_distance_ = GetGoalDistance();
}

void VehiclePlugin::Shutdown() {

	// Shutdown the simulation.
	gazebo::msgs::ServerControl msg;
	msg.set_stop(true);
	server_pub_->Publish(msg);
}
} // End of namespace gazebo.
