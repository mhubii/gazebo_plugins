#include "manual_control.h"

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

#define WORLD_NAME "vehicle_world"
#define VEHICLE_NAME "vehicle"
#define GOAL_COLLISION "goal::goal::goal_collision"
#define COLLISION_FILTER "ground_plane::link::collision"

namespace gazebo
{

VehiclePlugin::VehiclePlugin() :
	ModelPlugin(), multi_camera_node_(new gazebo::transport::Node()), collision_node_(new gazebo::transport::Node()) {

	vel_delta_ = 1e-3;

	reload_ = false;

	for (int i = 0; i < DOF; i++) {

		vel_[i] = 0;
	}

	// Initialize record to false.
	record_ = false;
	img_location_ = "";
	txt_location_ = "";

	// Set the start time.
	start_time_ = std::chrono::steady_clock::now();
	time_stamp_ = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time_);

	keyboard_ = Keyboard::Create();
}

void VehiclePlugin::Load(physics::ModelPtr parent, sdf::ElementPtr sdf) {

	// Set the start time.
	start_time_ = std::chrono::steady_clock::now();
	time_stamp_ = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time_);

	// Store the pointer to the model_.
	this->model_ = parent;

	// Get sdf parameters.
	if (!sdf->HasElement("record")) {

		record_ = false;
		img_location_ = "";
		txt_location_ = "";
	}
	else {

		record_ = sdf->Get<bool>("record");
		img_location_ = sdf->GetElement("record")->Get<std::string>("img_location");
		txt_location_ = sdf->GetElement("record")->Get<std::string>("txt_location");

		if (img_location_.empty() || txt_location_.empty()) {

			printf("VehicleManualControl -- please provide img_location and txt_location.");
			record_ = false;
		}

		if (record_) {

			printf("VehicleManualControl -- recording images to %s. \n", img_location_.c_str());
			printf("VehicleManualControl -- storing txt to %s. \n", txt_location_.c_str());
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

	// Create a node for camera communication.
	multi_camera_node_->Init();
	multi_camera_sub_ = multi_camera_node_->Subscribe("/gazebo/" WORLD_NAME "/" VEHICLE_NAME "/chassis/stereo_camera/images", &VehiclePlugin::OnCameraMsg, this);
	
	// Create a node for collision detection.
	collision_node_->Init();
	collision_sub_ = collision_node_->Subscribe("/gazebo/" WORLD_NAME "/" VEHICLE_NAME "/chassis/chassis_contact", &VehiclePlugin::OnCollisionMsg, this);

	// Listen to the update event. This event is broadcast every simulation iterartion.
	this->update_connection_ = event::Events::ConnectWorldUpdateBegin(std::bind(&VehiclePlugin::OnUpdate, this));
}

void VehiclePlugin::OnUpdate() {

	UpdateJoints();

	for(int i = 0; i < DOF; i++) {
		if(vel_[i] < VELOCITY_MIN)
			vel_[i] = VELOCITY_MIN;

		if(vel_[i] > VELOCITY_MAX)
			vel_[i] = VELOCITY_MAX;
	}

	if (joints_.size() != 8) {
		
		printf("VehicleManualControl -- could only find %zu of 8 drive joints\n", joints_.size());
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

	// Set current time.
	time_stamp_ = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time_);

	if (reload_) {

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

	if (!msg) {

		printf("VehicleManualControl -- received NULL message\n");
		return;
	}

	const int l_bpp = (msg->image()[0].step()/msg->image()[0].width())*8; // Bits per pixel.

	if (l_bpp != 24) {

		printf("VehicleManualControl -- expected 24 bits per pixel uchar3 image from camera, got %i\n", l_bpp);
		return;
	}

	const int r_bpp = (msg->image()[1].step()/msg->image()[0].width())*8; // Bits per pixel.

	if (r_bpp != 24) {

		printf("VehicleManualControl -- expected 24 bits per pixel uchar3 image from camera, got %i\n", r_bpp);
		return;
	}

	// Fill stringstream with preceeding zeros.
	std::ostringstream ss;
	ss << std::setw(8) << std::setfill('0') << std::to_string(time_stamp_.count());

	// Record images and states.
	if (record_) {

		std::ofstream binary(img_location_ + "/left/img" + ss.str() + ".raw", std::ios::out | std::ios::binary);
		binary.write((char*)msg->image()[0].data().c_str(), msg->image()[0].data().length());
		binary.close();

		binary.open(img_location_ + "/right/img" + ss.str() + ".raw", std::ios::out | std::ios::binary);
		binary.write((char*)msg->image()[1].data().c_str(), msg->image()[1].data().length());
		binary.close();

		std::ofstream txt(txt_location_ + "/log.txt", std::ios_base::app);

		txt << img_location_ + "/left/img" + ss.str() + ".raw" + ", ";
		txt << img_location_ + "/right/img" + ss.str() + ".raw";
		
		for (int i = 0; i < DOF; i++) {

			txt << ", " << vel_[i];
		}

		txt << "\n";
	}
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

		reload_ = (contacts->contact(i).collision1().compare(GOAL_COLLISION) == 0||
		           contacts->contact(i).collision2().compare(GOAL_COLLISION) == 0);
	}
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

	printf("VehicleManualControl -- failed to find joint '%s'\n", name);
	return false;
}

void VehiclePlugin::UpdateJoints() {

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

			vel_[i] = 0;
		}

	}
}
} // End of namespace gazebo.
