#include "vehicle_plugin.h"

#define L_FRONT_PITCH "vehicle::l_front_wheel_pitch"
#define L_FRONT_ROLL "vehicle::l_front_wheel_roll"
#define R_FRONT_PITCH "vehicle::r_front_wheel_pitch"
#define R_FRONT_ROLL "vehicle::r_front_wheel_roll"
#define L_BACK_PITCH "vehicle::l_back_wheel_pitch"
#define L_BACK_ROLL "vehicle::l_back_wheel_roll"
#define R_BACK_PITCH "vehicle::r_back_wheel_pitch"
#define R_BACK_ROLL "vehicle::r_back_wheel_roll"

#define VELOCITY_MIN -10.0f
#define VELOCITY_MAX  10.0f

#define WORLD_NAME "default" // To change later.
#define VEHICLE_NAME "vehicle"
#define GOAL_NAME "goal"
#define COLLISION_FILTER "ground_plane::link::collision"

namespace gazebo
{

//TMP
struct Options {

    std::string data_root{"data"};
    int32_t batch_size{64};
    int32_t epochs{10};
    double lr{0.01};
    double momentum{0.5};
    bool no_cuda{false};
    int32_t seed{1};
    int32_t test_batch_size{1000};
    int32_t log_interval{10};
};


VehiclePlugin::VehiclePlugin() :
	ModelPlugin(), multi_camera_node_(new gazebo::transport::Node()), collision_node_(new gazebo::transport::Node()), device_(torch::kCUDA) {

	op_mode_   = USER_MANUAL;
	new_state_ = false;
	vel_delta_ = 1e-3;

	for (int i = 0; i < DOF; i++) {
	
		vel_[i] = 0.;
	}

	keyboard_ = Keyboard::Create();

	//TMP
    Options options;

	model_.to(device_);
}

void VehiclePlugin::Load(physics::ModelPtr parent, sdf::ElementPtr /*sdf*/) {

	// Store the pointer to the model.
	this->model = parent;

	// Configure the joints.
	ConfigureJoints(L_FRONT_PITCH);
	ConfigureJoints(R_FRONT_PITCH);
	ConfigureJoints(L_BACK_PITCH);
	ConfigureJoints(R_BACK_PITCH);

	ConfigureJoints(L_FRONT_ROLL);
	ConfigureJoints(R_FRONT_ROLL);
	ConfigureJoints(L_BACK_ROLL);
	ConfigureJoints(R_BACK_ROLL);

	// Create Q-Learning agent.
	if (!CreateAgent()) {

		printf("VehiclePlugin -- failed to create Q-Learning agent\n");
	}

	// Create a node for camera communication.
	multi_camera_node_->Init();
	multi_camera_sub_ = multi_camera_node_->Subscribe("/gazebo/" WORLD_NAME "/" VEHICLE_NAME "/chassis/stereo_camera/images", &VehiclePlugin::OnCameraMsg, this);
	
	// Create a node for collision detection.
	collision_node_->Init();
	collision_sub_ = collision_node_->Subscribe("/gazebo/" WORLD_NAME "/" VEHICLE_NAME "/chassis/chassis_contact", &VehiclePlugin::OnCollisionMsg, this);

	// Listen to the update event. This event is broadcast every simulation iterartion.
	this->update_connection = event::Events::ConnectWorldUpdateBegin(std::bind(&VehiclePlugin::OnUpdate, this));
}

void VehiclePlugin::OnUpdate() {

	if (UpdateJoints()) {

		for(int i = 0; i < DOF; i++) {
			if(vel_[i] < VELOCITY_MIN)
				vel_[i] = VELOCITY_MIN;

			if(vel_[i] > VELOCITY_MAX)
				vel_[i] = VELOCITY_MAX;
		}

		if (joints_.size() != 8) {
			
			printf("VehiclePlugin -- could only find %zu of 8 drive joints\n", joints_.size());
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
	}
}

void VehiclePlugin::OnCameraMsg(ConstImagesStampedPtr &msg) {

	if (!msg) {

		printf("VehiclePlugin -- received NULL message\n");
		return;
	}

	const int l_width = msg->image()[0].width();
	const int l_height = msg->image()[0].height();
	const int l_bpp = (msg->image()[0].step()/msg->image()[0].width())*8; // Bits per pixel.
	const int l_size = msg->image()[0].data().size();

	if (l_bpp != 24) {

		printf("VehiclePlugin -- expected 24 bits per pixel uchar3 image from camera, got %i\n", l_bpp);
		return;
	}

	torch::Tensor l_img = torch::zeros({1, 3, l_height, l_width}, device_);

	// memcpy(l_img.data_ptr(), msg->image()[0].data().c_str(), l_size);

	// cv::Mat cv_l_img(l_height, l_width, CV_8UC3);
	// memcpy(cv_l_img.data, l_img.data_ptr(), l_size);

	//std::cout << cv_l_img << std::endl;

	model_.forward(l_img);

	const int r_width = msg->image()[1].width();
	const int r_height = msg->image()[1].height();
	const int r_bpp = (msg->image()[1].step()/msg->image()[0].width())*8; // Bits per pixel.
	const int r_size = msg->image()[1].data().size();

	if (r_bpp != 24) {

		printf("VehiclePlugin -- expected 24 bits per pixel uchar3 image from camera, got %i\n", r_bpp);
		return;
	}

	cv::Mat r_img(r_height, r_width, CV_8UC3);
	memcpy(r_img.data, msg->image()[1].data().c_str(), r_size);	
	
	new_state_ = true;
}

void VehiclePlugin::OnCollisionMsg(ConstContactsPtr &contacts)
{
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

		// issue learning reward
		// if( opMode == AGENT_LEARN )
		// {
		// 	#define GOAL_COLLISION "goal::link::box_collision"

		// 	bool hitTarget = (strcmp(contacts->contact(i).collision2().c_str(), GOAL_COLLISION) == 0) ||
		// 				  (strcmp(contacts->contact(i).collision1().c_str(), GOAL_COLLISION) == 0);

		// 	rewardHistory = hitTarget ? REWARD_WIN : REWARD_LOSS;

		// 	newReward  = true;
		// 	endEpisode = true;
		// }
	}
}

bool VehiclePlugin::CreateAgent() {

	return true;
}


bool VehiclePlugin::UpdateAgent() {

	return true;
}

bool VehiclePlugin::ConfigureJoints(const char* name) {

	std::vector<physics::JointPtr> joints = model->GetJoints();
	const size_t num_joints = joints.size();

	for (int i = 0; i < num_joints; i++) {

		if (strcmp(name, joints[i]->GetScopedName().c_str()) == 0) {
			
			joints[i]->SetVelocity(0, 0);
			joints_.push_back(joints[i]);
			return true;
		}
	}

	printf("VehiclePlugin -- failed to find joint '%s'\n", name);
	return false;
}

bool VehiclePlugin::UpdateJoints() {

	if (op_mode_ == USER_MANUAL) {

		keyboard_->Poll();
		
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

		return true;
	}
	else if (op_mode_ == AUTONOMOUSLY) {


		// No new processed state.
		new_state_ = false;

		if (UpdateAgent()) {
			
			return true;
		}
	}

	return false;
}
} // End of namespace gazebo.
