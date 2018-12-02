#include "autonomous_control.h"
#include <opencv4/opencv2/core.hpp>

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

	keyboard_ = Keyboard::Create();

    autonomous_ = false;
    new_state_ = false;

	l_img_ = torch::zeros({}, torch::kUInt8);
	r_img_ = torch::zeros({}, torch::kUInt8);
}

void VehiclePlugin::Load(physics::ModelPtr parent, sdf::ElementPtr sdf) {

	// Store the pointer to the model_.
	this->model_ = parent;

	// Get sdf parameters.
	if (sdf->HasElement("autonomous")) {

        autonomous_ = sdf->Get<bool>("autonomous");

		std::string script_module_location = sdf->GetElement("autonomous")->Get<std::string>("script_module");

		if (script_module_location.empty()) {

			printf("VehicleManualControl -- please provide a script_module.");
		}

        module_ = torch::jit::load(script_module_location);

		printf("VehicleManualControl -- successfully loaded scipt module from %s. \n", script_module_location.c_str());
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

    if (new_state_) {
	
        UpdateJoints();
        new_state_ = false;
    }

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

	const int l_width = msg->image()[0].width();
	const int l_height = msg->image()[0].height();
	const int l_size = msg->image()[0].data().size();
	const int l_bpp = (msg->image()[0].step()/msg->image()[0].width())*8; // Bits per pixel.

	if (l_bpp != 24) {

		printf("VehicleManualControl -- expected 24 bits per pixel uchar3 image from camera, got %i\n", l_bpp);
		return;
	}

	if (l_img_.sizes() != torch::IntList({1, 3, l_height, l_width})) {

		l_img_.resize_({1, 3, l_height, l_width});
	}

	const int r_width = msg->image()[1].width();
	const int r_height = msg->image()[1].height();
	const int r_size = msg->image()[1].data().size();
	const int r_bpp = (msg->image()[1].step()/msg->image()[0].width())*8; // Bits per pixel.

	if (r_bpp != 24) {

		printf("VehicleManualControl -- expected 24 bits per pixel uchar3 image from camera, got %i\n", r_bpp);
		return;
	}

	if (r_img_.sizes() != torch::IntList({1, 3, r_height, r_width})) {

		r_img_.resize_({1, 3, r_height, r_width});
	}

    // Copy image to tensor.
    std::memcpy(l_img_.data_ptr(), msg->image()[0].data().c_str(), l_size);
    std::memcpy(r_img_.data_ptr(), msg->image()[1].data().c_str(), r_size);

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

    if (autonomous_) {

        // Create a vector of inputs.
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(l_img_.to(torch::kFloat32));
        inputs.push_back(r_img_.to(torch::kFloat32));

        // Execute the model and turn its output into a tensor.
        torch::Tensor output = module_->forward(inputs).toTensor();
		
		for (int i = 0; i < DOF; i++) {

			vel_[i] = *(output.data<float>() + i);
		}
    }
    else {

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
}
} // End of namespace gazebo.
