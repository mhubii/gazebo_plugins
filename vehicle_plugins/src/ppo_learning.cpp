#include "ppo_learning.h"

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
	ppo_steps_ = 2560;           
	mini_batch_size_ = 1280;  
	ppo_epochs_ = 8;       
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

	initialized_ = false;
	new_state_ = false;
	state_updated_ = true;
	reset_ = false;

	reload_ = false;
	n_episodes_ = 0;
	n_steps_ = 0;

	final_state_ = NONE;

	height_ = 0; // set in VehiclePlugin::Load
	width_ = 0;

	c_ = 0;

	score_ = 0.;

	last_goal_distance_ = 0;
	goal_distance_reward_ = 0;
	hit_ = 0;

	last_left_state_ = torch::zeros({}, torch::kUInt8);
	last_right_state_ = torch::zeros({}, torch::kUInt8);
	last_action_ = torch::zeros({1,N_ACTIONS}, torch::kF32);
	last_value_ = torch::zeros({1,1}, torch::kF32);

	// // Next states.
	// l_img_next_ = torch::zeros({}, torch::kUInt8);
	// r_img_next_ = torch::zeros({}, torch::kUInt8);

	// dones_ = torch::zeros({1, 1}, torch::kInt);
}

VehiclePlugin::~VehiclePlugin() {

	delete opt_;

	if (track_) {

		out_file_vehicle_.close();	
		out_file_others_.close();	
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

	if (model_sdf->HasElement("sensor")) {

		model_sdf = model_sdf->GetElement("sensor");

		while (model_sdf != nullptr) {

			if (model_sdf->HasElement("camera")) {

				height_ = model_sdf->GetElement("camera")->GetElement("image")->Get<int>("height");
				width_ = model_sdf->GetElement("camera")->GetElement("image")->Get<int>("width");
				break;
			}

			model_sdf = model_sdf->GetNextElement("sensor");
		}
	}

	last_left_state_.resize_({1, height_, width_, 3});
	last_right_state_.resize_({1, height_, width_, 3});

	printf("VehiclePPOLearning -- got an input image of size %dx%d.\n", (int)height_, (int)width_);

	// Get sdf parameters.
	if (sdf->HasElement("settings")) {

		std::string mode = sdf->GetElement("settings")->Get<std::string>("mode");

		if (mode.empty()) {

			printf("VehiclePPOLearning -- please provide a mode, either train or test.\n");
		}
		else if (!std::strcmp(mode.c_str(), "test")) {

			printf("VehiclePPOLearning -- running in test mode.\n");
			train_ = false;
		}
		else if (!std::strcmp(mode.c_str(), "train")) {

			printf("VehiclePPOLearning -- running in train mode.\n");
			train_ = true;
		}

		// Optimization parameters.
		ppo_steps_ = sdf->GetElement("settings")->Get<int>("ppo_steps");
		mini_batch_size_ = sdf->GetElement("settings")->Get<int>("mini_batch_size");
		ppo_epochs_ = sdf->GetElement("settings")->Get<int>("ppo_epochs");
		max_episodes_  = sdf->GetElement("settings")->Get<int>("max_episodes");
		max_steps_ = sdf->GetElement("settings")->Get<int>("max_steps");

		reward_win_ = sdf->GetElement("settings")->Get<float>("reward_win");
		reward_loss_ = sdf->GetElement("settings")->Get<float>("reward_loss");
		cost_step_ = sdf->GetElement("settings")->Get<float>("cost_step");
		reward_goal_factor_ = sdf->GetElement("settings")->Get<float>("reward_goal_factor");

		randomness_ = sdf->GetElement("settings")->Get<bool>("randomness");

		printf("VehiclePPOLearning -- successfully initialized reinforcement learning in %s mode. \n", mode.c_str());
	}

	if (sdf->HasElement("track")) {

		track_ = sdf->Get<bool>("track");

		location_ = sdf->GetElement("track")->Get<std::string>("location");

		if (track_ && location_.empty()) {

			printf("VehiclePPOLearning -- please provide a location to store the tracked trajectory.");
			track_ = false;
		}
		else if (track_ && !location_.empty()) {

			out_file_vehicle_.open(location_ + "/vehicle_positions.csv");
			out_file_others_.open(location_ + "/goal_obstacle_positions.csv");
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
	printf("VehiclePPOLearning -- creating autonomous agent...\n");
	ac_ = ActorCritic(3, height_, width_, N_ACTIONS, 1e-2); // TODO to cuda?
	ac_->normal(0., 1e-2);
	ac_->to(torch::kF32);
	ac_->to(torch::kCUDA);
	opt_ = new torch::optim::Adam(ac_->parameters(), torch::optim::AdamOptions(1e-3));
	printf("VehiclePPOLearning -- successfully initialized agent.\n");

	if (sdf->HasElement("prior")) {

		// Load a prior policy.
		prior_ = sdf->Get<bool>("prior");

		std::string location = sdf->GetElement("prior")->Get<std::string>("location");

		if (prior_ && location.empty()) {

			printf("VehiclePPOLearning -- please provide a location with initial network parameters.\n");
			prior_ = false;
		}
		else if (prior_ && !location.empty()) {

			printf("VehiclePPOLearning -- setting prior network parameters.\n");
			torch::load(ac_, location + "/net.pt");
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

	start_time_ = time_.Float();
}

void VehiclePlugin::OnUpdate() {

	if (reset_) {

		ResetEnvironment();

		new_state_ = false;
		state_updated_ = true;
	}
	else if (new_state_) { // removed it from OnCameraMsg to this place

		auto av = ac_->forward(last_left_state_.to(torch::kF32).div(127.5).sub(1.).transpose(1, 3).transpose(2, 3).to(torch::kCUDA), 
					           last_right_state_.to(torch::kF32).div(127.5).sub(1.).transpose(1, 3).transpose(2, 3).to(torch::kCUDA));

		// Save actions and value.
		last_action_ = std::get<0>(av).to(torch::kCPU);
		last_value_ = std::get<1>(av).to(torch::kCPU);

		UpdateJoints(last_action_);

		n_steps_ += 1;

		new_state_ = false;
		state_updated_ = true;
	}
}

void VehiclePlugin::OnCameraMsg(ConstImagesStampedPtr &msg) {

	// Entry point.
	if (state_updated_) {

		if (track_) {

			// Track the position of the vehicle.
			ignition::math::Vector3d pos_vehicle = this->model_->GetWorld()->ModelByName("vehicle")->WorldPose().Pos();
			ignition::math::Vector3d pos_obstacle = this->model_->GetWorld()->ModelByName("obstacle")->WorldPose().Pos();
			ignition::math::Vector3d pos_goal = this->model_->GetWorld()->ModelByName("goal")->WorldPose().Pos();

			out_file_vehicle_ << pos_vehicle[0]  << ", " << pos_vehicle[1]  << ", " << pos_vehicle[2]  << "\n";
		}

		// Get the state of the environment.
		torch::Tensor l_img = torch::zeros({1, height_, width_, 3}, torch::kUInt8);
		torch::Tensor r_img = torch::zeros({1, height_, width_, 3}, torch::kUInt8);

		if (!MsgToTensor(msg, l_img, r_img)) {

			printf("VehiclePPOLearning -- could not convert message to tensor.\n");
		};
		
		state_updated_ = false;
		new_state_ = true;

		if (!initialized_) {
			last_goal_distance_ = GetGoalDistance();
			initialized_ = true;
		}
		else {

			// Do learning.
			left_states_.push_back(last_left_state_.to(torch::kF32));
			right_states_.push_back(last_right_state_.to(torch::kF32));

			actions_.push_back(last_action_);
			values_.push_back(last_value_);
			log_probs_.push_back(ac_->log_prob(last_action_.to(torch::kCUDA)).to(torch::kCPU));

			auto rd = Reward();

			rewards_.push_back(std::get<0>(rd));
			dones_.push_back(std::get<1>(rd));

			PrintStatus();

			if (*(dones_[c_].data<float>()) == 1) {

				reset_ = true;
			}

			c_++;

			if (c_%ppo_steps_ == 0) {

				printf("VehiclePPOLearning -- pausing the world while updating ppo.\n");
				gazebo::physics::pause_world(this->model_->GetWorld(), true);

				values_.push_back(std::get<1>(ac_->forward(last_left_state_.to(torch::kF32).div(127.5).sub(1.).transpose(1, 3).transpose(2, 3).to(torch::kCUDA), 
														   last_right_state_.to(torch::kF32).div(127.5).sub(1.).transpose(1, 3).transpose(2, 3).to(torch::kCUDA))).to(torch::kCPU));

				returns_ = PPO::returns(rewards_, dones_, values_, .99, .95);

				torch::Tensor t_log_probs = torch::cat(log_probs_).detach().to(torch::kCUDA);
				torch::Tensor t_returns = torch::cat(returns_).detach().to(torch::kCUDA);
				torch::Tensor t_values = torch::cat(values_).detach().to(torch::kCUDA);
				torch::Tensor t_left_states = torch::cat(left_states_).unsqueeze(1).to(torch::kF32).div(127.5).sub(1.).transpose(2, 4).transpose(3, 4).to(torch::kCUDA);
				torch::Tensor t_right_states = torch::cat(right_states_).unsqueeze(1).to(torch::kF32).div(127.5).sub(1.).transpose(2, 4).transpose(3, 4).to(torch::kCUDA);
				torch::Tensor t_actions = torch::cat(actions_).to(torch::kCUDA);
				torch::Tensor t_advantages = t_returns - t_values.slice(0, 0, ppo_steps_);

				PPO::update(ac_, t_left_states, t_right_states, t_actions, t_log_probs, t_returns, t_advantages, *opt_, ppo_steps_, ppo_epochs_, mini_batch_size_);

				c_ = 0;

				left_states_.clear();
				right_states_.clear();
				actions_.clear();
				rewards_.clear();
				dones_.clear();

				log_probs_.clear();
				returns_.clear();
				values_.clear();

				gazebo::physics::pause_world(this->model_->GetWorld(), false);
				printf("VehiclePPOLearning -- updating done, running world again.\n");
			}
		}

		// Store states.
		last_left_state_ = l_img;
		last_right_state_ = r_img;
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
			  << "    action: "    << *(actions_[c_].data<float>())
			  << "    dones: "     << *(dones_[c_].data<float>())
			  << "    reward: "    << *(rewards_[c_].data<float>()) << std::endl;
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

	printf("VehiclePPOLearning -- failed to find joint '%s'\n", name);
	return false;
}

bool VehiclePlugin::MsgToTensor(ConstImagesStampedPtr& msg, torch::Tensor& l_img, torch::Tensor& r_img) {

	if (!msg) {

		printf("VehiclePPOLearning -- received NULL message.\n");
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

	if (l_img.sizes() != torch::IntList({1, l_height, l_width, 3})) {

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

	if (r_img.sizes() != torch::IntList({1, r_height, r_width, 3})) {

		printf("VehicleAReinforcementLearning -- resizing tensor to %ix%ix%ix%i", 1, l_height, l_width, 3);
		r_img.resize_({1, r_height, r_width, 3});
	}

	// Copy image to tensor.
	std::memcpy(l_img.data_ptr(), msg->image().Get(0).data().c_str(), l_size);
	std::memcpy(r_img.data_ptr(), msg->image().Get(1).data().c_str(), r_size);

	return true;
}

auto VehiclePlugin::Reward() -> std::tuple<torch::Tensor, torch::Tensor> {
		
	// Determine the reward.
	float goal_distance = GetGoalDistance();
	goal_distance_reward_ = last_goal_distance_ - goal_distance;
	last_goal_distance_ = goal_distance;

	float cost = reward_goal_factor_*goal_distance_reward_ - cost_step_*float(n_steps_);
	torch::Tensor reward = torch::full({1, 1}, cost, torch::kF32);
	torch::Tensor done = torch::zeros({1,1}, torch::kF32);

	if (goal_distance < 0.9) {

		final_state_ = HIT_GOAL;
		reset_ = true;
		done[0][0] = 1;
		reward[0][0] += reward_win_;
		printf("VehiclePPOLearning -- hit goal.\n");
	}
	if (GetObstacleDistance() < 0.55) {

		final_state_ = HIT_OBSTACLE;
		reset_ = true;
		done[0][0] = 1;
		reward[0][0] += reward_loss_;
		printf("VehiclePPOLearning -- hit obstacle.\n");
	}
	if (max_steps_ <= n_steps_) {

		final_state_ = MAXIMUM_STEPS;
		reset_ = true;
		done[0][0] = 1;
		printf("VehiclePPOLearning -- maximum steps reached.\n");
	}
	if (final_state_ == NONE) {
		done[0][0] = 0;
	}

	return std::make_tuple(reward, done);
}

void VehiclePlugin::UpdateJoints(torch::Tensor& vel) {

	// Perform an action, given the new velocity.
	// Drive forward/backward and turn.
	joints_[0]->SetVelocity(0, *(vel.data<float>())); // left
	joints_[1]->SetVelocity(0, *(vel.data<float>()+1)); // right
	joints_[2]->SetVelocity(0, *(vel.data<float>())); // left
	joints_[3]->SetVelocity(0, *(vel.data<float>()+1)); // right

	// Drive left/right. Rotate to frames.
	ignition::math::Vector3<double> axis = axis.UnitX;
	ignition::math::Vector3<double> tmp = tmp.Zero;

	ignition::math::Quaterniond ori = joints_[0]->AxisFrameOffset(0);
	tmp = ori.RotateVector(axis);
	joints_[4]->SetAxis(0, tmp);
	joints_[4]->SetVelocity(0, *(vel.data<float>()+2));

	ori = joints_[1]->AxisFrameOffset(0);		
	tmp = ori.RotateVector(axis);
	joints_[5]->SetAxis(0, tmp);
	joints_[5]->SetVelocity(0, *(vel.data<float>()+2));

	ori = joints_[2]->AxisFrameOffset(0);
	tmp = ori.RotateVector(axis);
	joints_[6]->SetAxis(0, tmp);
	joints_[6]->SetVelocity(0, *(vel.data<float>()+2));

	ori = joints_[3]->AxisFrameOffset(0);
	tmp = ori.RotateVector(axis);
	joints_[7]->SetAxis(0, tmp);
	joints_[7]->SetVelocity(0, *(vel.data<float>()+2));
}

void VehiclePlugin::ResetEnvironment() {

	if (track_) {

		// Track the position of the goal and the obstacle.
		ignition::math::Vector3d pos_obstacle = this->model_->GetWorld()->ModelByName("obstacle")->WorldPose().Pos();
		ignition::math::Vector3d pos_goal = this->model_->GetWorld()->ModelByName("goal")->WorldPose().Pos();

		out_file_others_ << pos_goal[0]     << ", " << pos_goal[1]     << ", " << pos_goal[2]     << ", "
				         << pos_obstacle[0] << ", " << pos_obstacle[1] << ", " << pos_obstacle[2] << "\n";

		end_time_ = time_.Float();

		start_time_ = end_time_;

		// // Save neural net on best mean loss.
		// if (mean_score > best_score_) {

		// 	torch::save(ac_, location_ + "/net.pt");
		// 	best_score_ = mean_score;
		// }
	}

	reset_ = false;
	initialized_ = false;

	n_episodes_ += 1;
	n_steps_ = 0;
	final_state_ = NONE;

	score_ = 0.;

	if (max_episodes_ <= n_episodes_) {

		// Shutdown simulation.
		printf("VehiclePPOLearning -- maximal episodes reached, shutting down.\n");
		Shutdown();
	}

	// End episode.
	printf("VehiclePPOLearning -- resetting agent.\n");

	// Reset environment.
	for (int i = 0; i < DOF; i++) {

		vel_[i] = 0.;
	}

	torch::Tensor vel = torch::zeros({1,3}, torch::kF32);
	UpdateJoints(vel);

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
}

void VehiclePlugin::Shutdown() {

	// Shutdown the simulation.
	gazebo::msgs::ServerControl msg;
	msg.set_stop(true);
	server_pub_->Publish(msg);
}
} // End of namespace gazebo.
