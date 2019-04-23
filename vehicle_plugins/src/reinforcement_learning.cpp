#include "reinforcement_learning.h"

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

#define BATCH_SIZE 64
#define BUFFER_SIZE 2560
#define MAX_EPISODES 100
#define MAX_STEPS 600

#define REWARD_WIN  1000
#define REWARD_LOSS -1000
#define REWARD_STEP -0.01

#define WORLD_NAME "vehicle_world"
#define VEHICLE_NAME "vehicle"
#define GOAL_COLLISION "goal::goal::goal_collision"
#define COLLISION_FILTER "ground_plane::link::collision"

namespace gazebo
{

VehiclePlugin::VehiclePlugin() :
	ModelPlugin(), node_(new gazebo::transport::Node()) {

	reload_ = false;

	n_episodes_ = 0;
	n_steps_ = 0;

	autonomous_ = false;
	new_state_ = false;
	vel_delta_ = 1e-3;

	for (int i = 0; i < DOF; i++) {
	
		vel_[i] = 0.;
	}

	// States.
	l_img_ = torch::zeros({}, torch::kUInt8);
	r_img_ = torch::zeros({}, torch::kUInt8);

	// Actions and rewards.
	action_ = torch::zeros({1, 3}, torch::kFloat32);
	reward_ = torch::zeros({1, 1}, torch::kFloat32);

	// Next states.
	l_img_next_ = torch::zeros({}, torch::kUInt8);
	r_img_next_ = torch::zeros({}, torch::kUInt8);

	// Dones.
	dones_ = torch::zeros({1, 1}, torch::kInt);
}

VehiclePlugin::~VehiclePlugin() {

	if (autonomous_) {

		delete brain_;
	}
}

void VehiclePlugin::Load(physics::ModelPtr parent, sdf::ElementPtr sdf) {

	// Store the pointer to the model.
	this->model_ = parent;

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

	printf("VehicleReinforcementLearning -- got an input image of size %dx%d.\n", (int)height, (int)width);

	// Get sdf parameters.
	if (sdf->HasElement("autonomous")) {

        autonomous_ = sdf->Get<bool>("autonomous");

		std::string mode = sdf->GetElement("autonomous")->Get<std::string>("mode");

		if (mode.empty()) {

			printf("VehicleReinforcementLearning-- please provide a mode, either train or test.");
		}

		if (autonomous_) {
		
			printf("VehicleReinforcementLearning -- successfully initialized reinforcement learning in %s mode. \n", mode.c_str());
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
	
		printf("VehicleReinforcementLearning -- creating autonomous agent...\n");
		brain_ = new DDPGContinuousControl({3, height, width}, DOF, BATCH_SIZE, BUFFER_SIZE, torch::kCUDA);
		printf("VehicleReinforcementLearning -- successfully initialized agent.\n");

		brain_->Reset();
	}

	// Node for communication.
	node_->Init();

	// Create a node for camera communication.
	multi_camera_sub_ = node_->Subscribe("/gazebo/" WORLD_NAME "/" VEHICLE_NAME "/chassis/stereo_camera/images", &VehiclePlugin::OnCameraMsg, this);

	// Create a node for collision detection.
	collision_sub_ = node_->Subscribe("/gazebo/" WORLD_NAME "/" VEHICLE_NAME "/chassis/chassis_contact", &VehiclePlugin::OnCollisionMsg, this);

	// Create a node for server communication.
	server_pub_ = node_->Advertise<gazebo::msgs::ServerControl>("/gazebo/server/control");

	// Listen to the update event. This event is broadcast every simulation iterartion.
	this->update_connection = event::Events::ConnectWorldUpdateBegin(std::bind(&VehiclePlugin::OnUpdate, this));

	keyboard_ = Keyboard::Create();

	if (!keyboard_) {

		printf("VehicleReinforcementLearning -- no keyboard for manual control, shutting down.\n");
		Shutdown();
	}
}

void VehiclePlugin::OnUpdate() {

	if(!UpdateJoints()) {

		printf("VehicleReinforcementLearning -- failed to update the agent.\n");
	}

	// for(int i = 0; i < DOF; i++) {
	// 	if(vel_[i] < VELOCITY_MIN)
	// 		vel_[i] = VELOCITY_MIN;

	// 	if(vel_[i] > VELOCITY_MAX)
	// 		vel_[i] = VELOCITY_MAX;
	// }

	if (joints_.size() != 8) {
		
		printf("VehicleReinforcementLearning -- could only find %zu of 8 drive joints\n", joints_.size());
		return;
	}

	// Drive forward/backward and turn.
	joints_[0]->SetVelocity(0, vel_[0]); // left
	joints_[1]->SetVelocity(0, vel_[1]); // right
	joints_[2]->SetVelocity(0, vel_[0]); // left
	joints_[3]->SetVelocity(0, vel_[1]); // right

	// Drive left/right. Rotate to frames.
	ignition::math::Vector3<double> axis = axis.UnitX;
	ignition::math::Vector3<double> tmp = tmp.Zero;

	ignition::math::Quaterniond ori = joints_[0]->AxisFrameOffset(0);
	tmp = ori.RotateVector(axis);
	joints_[4]->SetAxis(0, tmp);
	joints_[4]->SetVelocity(0, vel_[2]);

	ori = joints_[1]->AxisFrameOffset(0);		
	tmp = ori.RotateVector(axis);
	joints_[5]->SetAxis(0, tmp);
	joints_[5]->SetVelocity(0, vel_[2]);

	ori = joints_[2]->AxisFrameOffset(0);
	tmp = ori.RotateVector(axis);
	joints_[6]->SetAxis(0, tmp);
	joints_[6]->SetVelocity(0, vel_[2]);

	ori = joints_[3]->AxisFrameOffset(0);
	tmp = ori.RotateVector(axis);
	joints_[7]->SetAxis(0, tmp);
	joints_[7]->SetVelocity(0, vel_[2]);

	if (reload_) {

		n_episodes_ += 1;
		n_steps_ = 0;

		// Reset rewards.
		hit_ = 0.;

		if (MAX_EPISODES <= n_episodes_) {

			// Shutdown simulation.
			printf("VehicleReinforcementLearning -- maximal episodes reached, shutting down.\n");
			Shutdown();
		}

		// End episode.
		printf("VehicleReinforcementLearning -- resetting agent.\n");
		brain_->Reset();

		// Reset environment.
		reload_ = false;

		for (int i = 0; i < DOF; i++) {

			vel_[i] = 0.;
		}

		model_->SetAngularVel(ignition::math::Vector3d(0., 0., 0.));
		model_->SetLinearVel(ignition::math::Vector3d(0., 0., 0.));

		// Set initial pose on goal hit.
		model_->SetRelativePose(init_pose_);
	}
}

void VehiclePlugin::OnCameraMsg(ConstImagesStampedPtr &msg) {

	n_steps_ += 1;

	if (MAX_STEPS <= n_steps_) {

		reload_ = true;
	}

	if (autonomous_) {

		if (!msg) {

			printf("VehicleReinforcementLearning -- received NULL message.\n");
			return;
		}

		const int l_width = msg->image().Get(0).width();
		const int l_height = msg->image().Get(0).height();
		const int l_size = msg->image().Get(0).data().size();
		const int l_bpp = (msg->image().Get(0).step()/msg->image().Get(0).width())*8; // Bits per pixel.

		if (l_bpp != 24) {

			printf("VehicleAReinforcementLearning -- expected 24 bits per pixel uchar3 image from camera, got %i.\n", l_bpp);
			return;
		}

		if (l_img_next_.sizes() != torch::IntArrayRef({1, l_height, l_width, 3})) {

			printf("VehicleAReinforcementLearning -- resizing tensor to %ix%ix%ix%i", 1, l_height, l_width, 3);
			l_img_next_.resize_({1, l_height, l_width, 3});
		}

		const int r_width = msg->image().Get(1).width();
		const int r_height = msg->image().Get(1).height();
		const int r_size = msg->image().Get(1).data().size();
		const int r_bpp = (msg->image().Get(1).step()/msg->image().Get(0).width())*8; // Bits per pixel.

		if (r_bpp != 24) {

			printf("VehicleAReinforcementLearning -- expected 24 bits per pixel uchar3 image from camera, got %i.\n", r_bpp);
			return;
		}

		if (r_img_next_.sizes() != torch::IntArrayRef({1, r_height, r_width, 3})) {

			printf("VehicleAReinforcementLearning -- resizing tensor to %ix%ix%ix%i", 1, l_height, l_width, 3);
			r_img_next_.resize_({1, r_height, r_width, 3});
		}

		// Copy image to tensor.
		std::memcpy(l_img_next_.data_ptr(), msg->image().Get(0).data().c_str(), l_size);
		std::memcpy(r_img_next_.data_ptr(), msg->image().Get(1).data().c_str(), r_size);

		new_state_ = true;

		// Get the goal distance corresponding to the current state.
		physics::ModelPtr goal = this->model_->GetWorld()->ModelByName("goal");
		
		ignition::math::Box goalbox = goal->BoundingBox();
		ignition::math::Box chassisbox = this->model_->GetLink("chassis")->BoundingBox();

		goal_distance_ = BoxDistance(chassisbox, goalbox);
	}
}

void VehiclePlugin::OnCollisionMsg(ConstContactsPtr &contacts) {
	
	for (unsigned int i = 0; i < contacts->contact_size(); ++i)
	{
		if( strcmp(contacts->contact(i).collision2().c_str(), COLLISION_FILTER) == 0 )
			continue;

		std::cout << "Collision between[" << contacts->contact(i).collision1()
			     << "] and [" << contacts->contact(i).collision2() << "]\n";


		for (unsigned int j = 0; j < contacts->contact(i).position_size(); ++j)
		{
			 std::cout << j << "  Position:"
					   << contacts->contact(i).position(j).x() << " "
					   << contacts->contact(i).position(j).y() << " "
					   << contacts->contact(i).position(j).z() << "\n";
			 std::cout << "   Normal:"
					   << contacts->contact(i).normal(j).x() << " "
					   << contacts->contact(i).normal(j).y() << " "
					   << contacts->contact(i).normal(j).z() << "\n";
			 std::cout << "   Depth:" << contacts->contact(i).depth(j) << "\n";
		}

		reload_ = (contacts->contact(i).collision1().compare(GOAL_COLLISION) == 0||
		           contacts->contact(i).collision2().compare(GOAL_COLLISION) == 0);

		// Set reward for hit.
		hit_ = reload_ ? REWARD_WIN : REWARD_LOSS;

		reload_ = true;
	}
}

bool VehiclePlugin::UpdateAgent() {

	// Set terminate state.
	dones_[0][0] = reload_ ? 1 : 0;

	// Set reward of performed action.
	reward_[0][0] = -10*goal_distance_ + hit_ + n_steps_*REWARD_STEP;
	
	if (n_steps_ % 100 == 0) {
			
		std::cout << "episode: "     << n_episodes_
				  << "    step: "    << n_steps_ 
			      << "    action: (" << *(action_.to(torch::kCPU).data<float>() + 0)*VELOCITY_MAX << ", "
			    					 << *(action_.to(torch::kCPU).data<float>() + 1)*VELOCITY_MAX << ", "
			    					 << *(action_.to(torch::kCPU).data<float>() + 2)*VELOCITY_MAX << ")"
			      << "    dones: "   << *(dones_.data<int>())
			      << "    reward: "  << *(reward_.data<float>()) << std::endl;
	}
	


	// Put states, actions, rewards, and everything together.
	state bundle{l_img_.to(torch::kFloat32).div(127.5).sub(1.).transpose(1, 3).transpose(2, 3)/*init*/,
	             r_img_.to(torch::kFloat32).div(127.5).sub(1.).transpose(1, 3).transpose(2, 3)/*init*/,
				 action_.to(torch::kFloat32)/*action*/,
				 reward_.to(torch::kFloat32)/*reward*/,
				 l_img_next_.to(torch::kFloat32).div(127.5).sub(1.).transpose(1, 3).transpose(2, 3)/*next*/,
				 r_img_next_.to(torch::kFloat32).div(127.5).sub(1.).transpose(1, 3).transpose(2, 3)/*next*/,
				 dones_.to(torch::kFloat32)/*dones*/};

	brain_->Step(bundle);

	// Perform an action.
	action_ = brain_->Act(l_img_next_.to(torch::kFloat32).div(127.5).sub(1.).transpose(1, 3).transpose(2, 3),
	                      r_img_next_.to(torch::kFloat32).div(127.5).sub(1.).transpose(1, 3).transpose(2, 3), true); // MEMORY LEAK TODO

	if (action_.numel() == 0) {

		return false;
	}

	for (int i = 0; i < DOF; i++) {

		vel_[i] = *(action_.to(torch::kCPU).data<float>() + i)*VELOCITY_MAX;
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

	printf("VehicleReinforcementLearning -- failed to find joint '%s'\n", name);
	return false;
}

bool VehiclePlugin::UpdateJoints() {

	keyboard_->Poll();

	if (autonomous_ && new_state_) {

		// No new processed state.
		new_state_ = false;

		if (!UpdateAgent()) {
			
			return false;
		}
	}

	else if (!autonomous_) {

		if (keyboard_->KeyDown(KEY_W)) {
			
			vel_[0] += vel_delta_;
			vel_[1] += vel_delta_;
		}
		if (keyboard_->KeyDown(KEY_S)) {
			
			vel_[0] -= vel_delta_;
			vel_[1] -= vel_delta_;
		}
		if (keyboard_->KeyDown(KEY_D)) {
			
			vel_[0] += vel_delta_;
			vel_[1] -= vel_delta_;
		}
		if (keyboard_->KeyDown(KEY_A)) {
			
			vel_[0] -= vel_delta_;
			vel_[1] += vel_delta_;
		}
		if (keyboard_->KeyDown(KEY_LEFT)) {
			
			vel_[2] -= vel_delta_;
		}
		if (keyboard_->KeyDown(KEY_RIGHT)) {
			
			vel_[2] += vel_delta_;
		}
		if (keyboard_->KeyDown(KEY_E)) {
			
			for (int i = 0; i < DOF; i++) {
	
				vel_[i] = 0.;
			}
		}
		if (keyboard_->KeyDown(KEY_Q)) {

			printf("VehicleManualControl -- interruption after key q was pressed, shutting down.\n");	
			Shutdown();
		}
	}

	return true;
}

void VehiclePlugin::Shutdown() {

	// Shutdown the simulation.
	gazebo::msgs::ServerControl msg;
	msg.set_stop(true);
	server_pub_->Publish(msg);
}
} // End of namespace gazebo.
